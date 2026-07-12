// Fused sparse linear layer for Apple Silicon (MPS), runtime-compiled Metal.
//
// Layer semantics (weight is out_features x in_features, k nonzeros):
//   out[b,i] = sum_{t : rows[t]==i}  values[t] * x[b, cols[t]]
//
// Three float32 kernels, int32 index buffers, NO float atomics (CSR/CSC layout
// gives every output element a single owning thread):
//   1. forward       (CSR)  grid B*out_features
//   2. grad_val             one thread per nonzero (serial) OR one threadgroup
//                           per nonzero with a shared-memory batch reduction
//   3. grad_x        (CSC)  grid B*in_features
//
// Optimizations vs. the first version:
//   WIN 1: no CPU-blocking waits. Every dispatch uses SyncType::COMMIT (flush,
//          don't wait); ordering is preserved by the serial MPS stream, and the
//          test forces a real sync by copying results to CPU. The backward
//          encodes grad_val + grad_x into ONE command buffer and commits once.
//   WIN 2: grad_val batch reduction is parallelized across a threadgroup.
//
// Toolchain recipe mirrors the proven mps_probe.mm.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

using at::native::mps::getMTLBufferStorage;

// forward (CSR) + grad_x (CSC), compiled into one library.
static const char* FWD_GX_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void sparse_forward(device const float* x        [[buffer(0)]],
                           device const float* val      [[buffer(1)]],
                           device const int*   crow     [[buffer(2)]],
                           device const int*   csr_col  [[buffer(3)]],
                           device float*       out      [[buffer(4)]],
                           constant uint&      B        [[buffer(5)]],
                           constant uint&      in_f     [[buffer(6)]],
                           constant uint&      out_f    [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= B * out_f) return;
    uint b = gid / out_f;
    uint i = gid % out_f;
    int start = crow[i];
    int end   = crow[i + 1];
    float acc = 0.0f;
    device const float* xb = x + (ulong)b * in_f;
    for (int t = start; t < end; ++t) {
        acc += val[t] * xb[csr_col[t]];
    }
    out[gid] = acc;
}

kernel void sparse_grad_x(device const float* grad_out     [[buffer(0)]],
                          device const float* val          [[buffer(1)]],
                          device const int*   ccol         [[buffer(2)]],
                          device const int*   csc_rowidx   [[buffer(3)]],
                          device const int*   csc_valperm  [[buffer(4)]],
                          device float*       grad_x       [[buffer(5)]],
                          constant uint&      B            [[buffer(6)]],
                          constant uint&      in_f         [[buffer(7)]],
                          constant uint&      out_f        [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= B * in_f) return;
    uint b = gid / in_f;
    uint j = gid % in_f;
    int start = ccol[j];
    int end   = ccol[j + 1];
    float acc = 0.0f;
    device const float* gb = grad_out + (ulong)b * out_f;
    for (int s = start; s < end; ++s) {
        acc += val[csc_valperm[s]] * gb[csc_rowidx[s]];
    }
    grad_x[gid] = acc;
}
)METAL";

// grad_val: serial (one thread per nonzero) and threadgroup (one threadgroup per
// nonzero, shared-memory tree reduction over the batch). Both in one library.
static const char* GRAD_VAL_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;

// grad_val[t] = sum_b grad_out[b, csr_rowidx[t]] * x[b, csr_col[t]]  (serial)
kernel void sparse_grad_val_serial(device const float* grad_out    [[buffer(0)]],
                                   device const float* x           [[buffer(1)]],
                                   device const int*   csr_rowidx  [[buffer(2)]],
                                   device const int*   csr_col     [[buffer(3)]],
                                   device float*       grad_val    [[buffer(4)]],
                                   constant uint&      B           [[buffer(5)]],
                                   constant uint&      in_f        [[buffer(6)]],
                                   constant uint&      out_f       [[buffer(7)]],
                                   constant uint&      K           [[buffer(8)]],
                                   uint t [[thread_position_in_grid]]) {
    if (t >= K) return;
    int r = csr_rowidx[t];
    int c = csr_col[t];
    float acc = 0.0f;
    for (uint b = 0; b < B; ++b) {
        acc += grad_out[(ulong)b * out_f + r] * x[(ulong)b * in_f + c];
    }
    grad_val[t] = acc;
}

// Same result, but one threadgroup per nonzero reduces the batch dim in shared
// memory. Dispatch K threadgroups of T (power-of-two, <= 256) threads.
kernel void sparse_grad_val_tg(device const float* grad_out    [[buffer(0)]],
                               device const float* x           [[buffer(1)]],
                               device const int*   csr_rowidx  [[buffer(2)]],
                               device const int*   csr_col     [[buffer(3)]],
                               device float*       grad_val    [[buffer(4)]],
                               constant uint&      B           [[buffer(5)]],
                               constant uint&      in_f        [[buffer(6)]],
                               constant uint&      out_f       [[buffer(7)]],
                               uint t   [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint T   [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    int r = csr_rowidx[t];
    int c = csr_col[t];
    float local = 0.0f;
    for (uint b = tid; b < B; b += T) {
        local += grad_out[(ulong)b * out_f + r] * x[(ulong)b * in_f + c];
    }
    shared[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) grad_val[t] = shared[0];
}
)METAL";

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                                const char* src, const char* fname) {
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
    TORCH_CHECK(lib, "Metal compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fname]];
    TORCH_CHECK(fn, "no Metal function named ", fname);
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(pso, "pipeline creation failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    return pso;
}

static id<MTLComputePipelineState> psoForward(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, FWD_GX_SRC, "sparse_forward");
    return p;
}
static id<MTLComputePipelineState> psoGradX(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, FWD_GX_SRC, "sparse_grad_x");
    return p;
}
static id<MTLComputePipelineState> psoGradValSerial(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, GRAD_VAL_SRC, "sparse_grad_val_serial");
    return p;
}
static id<MTLComputePipelineState> psoGradValTG(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, GRAD_VAL_SRC, "sparse_grad_val_tg");
    return p;
}

static inline NSUInteger tgSize(id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger cap = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    NSUInteger tg = MIN(cap, n);
    return tg > 0 ? tg : 1;
}

static inline void setBuf(id<MTLComputeCommandEncoder> enc, torch::Tensor t, NSUInteger idx) {
    [enc setBuffer:getMTLBufferStorage(t)
            offset:t.storage_offset() * t.element_size()
           atIndex:idx];
}

// ---- encode helpers (no commit; caller owns the command buffer + sync) ----

static void encForward(id<MTLComputeCommandEncoder> enc, id<MTLDevice> dev,
                       torch::Tensor x, torch::Tensor values, torch::Tensor crow,
                       torch::Tensor csr_col, torch::Tensor out,
                       uint32_t B, uint32_t in_f, uint32_t out_f) {
    id<MTLComputePipelineState> pso = psoForward(dev);
    [enc setComputePipelineState:pso];
    setBuf(enc, x, 0); setBuf(enc, values, 1); setBuf(enc, crow, 2);
    setBuf(enc, csr_col, 3); setBuf(enc, out, 4);
    [enc setBytes:&B length:sizeof(B) atIndex:5];
    [enc setBytes:&in_f length:sizeof(in_f) atIndex:6];
    [enc setBytes:&out_f length:sizeof(out_f) atIndex:7];
    NSUInteger n = (NSUInteger)B * out_f;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
}

static void encGradX(id<MTLComputeCommandEncoder> enc, id<MTLDevice> dev,
                     torch::Tensor grad_out, torch::Tensor values, torch::Tensor ccol,
                     torch::Tensor csc_rowidx, torch::Tensor csc_valperm, torch::Tensor grad_x,
                     uint32_t B, uint32_t in_f, uint32_t out_f) {
    id<MTLComputePipelineState> pso = psoGradX(dev);
    [enc setComputePipelineState:pso];
    setBuf(enc, grad_out, 0); setBuf(enc, values, 1); setBuf(enc, ccol, 2);
    setBuf(enc, csc_rowidx, 3); setBuf(enc, csc_valperm, 4); setBuf(enc, grad_x, 5);
    [enc setBytes:&B length:sizeof(B) atIndex:6];
    [enc setBytes:&in_f length:sizeof(in_f) atIndex:7];
    [enc setBytes:&out_f length:sizeof(out_f) atIndex:8];
    NSUInteger n = (NSUInteger)B * in_f;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
}

static void encGradVal(id<MTLComputeCommandEncoder> enc, id<MTLDevice> dev, bool tg,
                       torch::Tensor grad_out, torch::Tensor x, torch::Tensor csr_rowidx,
                       torch::Tensor csr_col, torch::Tensor grad_val,
                       uint32_t B, uint32_t in_f, uint32_t out_f, uint32_t K) {
    if (tg) {
        id<MTLComputePipelineState> pso = psoGradValTG(dev);
        [enc setComputePipelineState:pso];
        setBuf(enc, grad_out, 0); setBuf(enc, x, 1); setBuf(enc, csr_rowidx, 2);
        setBuf(enc, csr_col, 3); setBuf(enc, grad_val, 4);
        [enc setBytes:&B length:sizeof(B) atIndex:5];
        [enc setBytes:&in_f length:sizeof(in_f) atIndex:6];
        [enc setBytes:&out_f length:sizeof(out_f) atIndex:7];
        // one threadgroup per nonzero; T = smallest power of two >= B, capped at
        // 256 (matches shared[256]) so we don't launch idle threads / extra
        // barrier levels when B is small.
        NSUInteger cap = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        NSUInteger T = 32;
        while (T < (NSUInteger)B && T < cap) T <<= 1;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)K, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
    } else {
        id<MTLComputePipelineState> pso = psoGradValSerial(dev);
        [enc setComputePipelineState:pso];
        setBuf(enc, grad_out, 0); setBuf(enc, x, 1); setBuf(enc, csr_rowidx, 2);
        setBuf(enc, csr_col, 3); setBuf(enc, grad_val, 4);
        [enc setBytes:&B length:sizeof(B) atIndex:5];
        [enc setBytes:&in_f length:sizeof(in_f) atIndex:6];
        [enc setBytes:&out_f length:sizeof(out_f) atIndex:7];
        [enc setBytes:&K length:sizeof(K) atIndex:8];
        NSUInteger n = (NSUInteger)K;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
    }
}

// ---- forward (CSR): single kernel, COMMIT (no wait) ----
torch::Tensor sparse_forward(torch::Tensor x, torch::Tensor values,
                             torch::Tensor crow, torch::Tensor csr_col,
                             int64_t out_features) {
    TORCH_CHECK(x.device().is_mps(), "x must be MPS");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "float32 only");
    x = x.contiguous();
    values = values.contiguous();
    uint32_t B = static_cast<uint32_t>(x.size(0));
    uint32_t in_f = static_cast<uint32_t>(x.size(1));
    uint32_t out_f = static_cast<uint32_t>(out_features);
    auto out = torch::empty({(int64_t)B, (int64_t)out_f}, x.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        encForward(enc, device, x, values, crow, csr_col, out, B, in_f, out_f);
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    return out;
}

// ---- fused backward: grad_val + grad_x in ONE command buffer, commit once ----
// tg_reduce=true selects the threadgroup grad_val kernel (WIN 2).
std::vector<torch::Tensor> sparse_backward(torch::Tensor grad_out, torch::Tensor x,
                                           torch::Tensor values,
                                           torch::Tensor csr_rowidx, torch::Tensor csr_col,
                                           torch::Tensor ccol, torch::Tensor csc_rowidx,
                                           torch::Tensor csc_valperm,
                                           bool need_grad_val, bool need_grad_x,
                                           bool tg_reduce) {
    TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be MPS");
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    values = values.contiguous();
    uint32_t B = static_cast<uint32_t>(x.size(0));
    uint32_t in_f = static_cast<uint32_t>(x.size(1));
    uint32_t out_f = static_cast<uint32_t>(grad_out.size(1));
    uint32_t K = static_cast<uint32_t>(csr_col.numel());

    auto grad_val = torch::empty({(int64_t)K}, x.options());
    auto grad_x = torch::empty({(int64_t)B, (int64_t)in_f}, x.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        stream->endKernelCoalescing();
        id<MTLCommandBuffer> cb = stream->commandBuffer();
        if (need_grad_val) {
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            encGradVal(e1, device, tg_reduce, grad_out, x, csr_rowidx, csr_col,
                       grad_val, B, in_f, out_f, K);
            [e1 endEncoding];
        }
        if (need_grad_x) {
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            encGradX(e2, device, grad_out, values, ccol, csc_rowidx, csc_valperm,
                     grad_x, B, in_f, out_f);
            [e2 endEncoding];
        }
        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    return {grad_val, grad_x};
}

// ---- standalone backward kernels (separate commits) -- for isolating WIN 1 ----
torch::Tensor sparse_grad_val(torch::Tensor grad_out, torch::Tensor x,
                              torch::Tensor csr_rowidx, torch::Tensor csr_col,
                              bool tg_reduce) {
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    uint32_t B = static_cast<uint32_t>(x.size(0));
    uint32_t in_f = static_cast<uint32_t>(x.size(1));
    uint32_t out_f = static_cast<uint32_t>(grad_out.size(1));
    uint32_t K = static_cast<uint32_t>(csr_col.numel());
    auto grad_val = torch::empty({(int64_t)K}, x.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        encGradVal(enc, device, tg_reduce, grad_out, x, csr_rowidx, csr_col,
                   grad_val, B, in_f, out_f, K);
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    return grad_val;
}

torch::Tensor sparse_grad_x(torch::Tensor grad_out, torch::Tensor values,
                            torch::Tensor ccol, torch::Tensor csc_rowidx,
                            torch::Tensor csc_valperm, int64_t in_features) {
    grad_out = grad_out.contiguous();
    values = values.contiguous();
    uint32_t B = static_cast<uint32_t>(grad_out.size(0));
    uint32_t out_f = static_cast<uint32_t>(grad_out.size(1));
    uint32_t in_f = static_cast<uint32_t>(in_features);
    auto grad_x = torch::empty({(int64_t)B, (int64_t)in_f}, grad_out.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        encGradX(enc, device, grad_out, values, ccol, csc_rowidx, csc_valperm,
                 grad_x, B, in_f, out_f);
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_forward", &sparse_forward, "fused sparse forward (CSR)");
    m.def("sparse_backward", &sparse_backward, "fused sparse backward (grad_val + grad_x, one commit)");
    m.def("sparse_grad_val", &sparse_grad_val, "sparse backward grad_val (standalone)");
    m.def("sparse_grad_x", &sparse_grad_x, "sparse backward grad_x (standalone, CSC)");
}
