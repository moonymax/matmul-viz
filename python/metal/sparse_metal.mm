// Fused sparse linear layer for Apple Silicon (MPS), runtime-compiled Metal.
//
// Layer semantics (weight is out_features x in_features, k nonzeros):
//   out[b,i] = sum_{t : rows[t]==i}  values[t] * x[b, cols[t]]
//
// Three float32 kernels, int32 index buffers, NO float atomics (CSR/CSC layout
// gives every output element a single owning thread):
//   1. forward   (CSR)  grid B*out_features
//   2. grad_val         grid k
//   3. grad_x    (CSC)  grid B*in_features
//
// Toolchain recipe mirrors the proven mps_probe.mm.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

using at::native::mps::getMTLBufferStorage;

// grad_val[t] = sum_b grad_out[b, csr_rowidx[t]] * x[b, csr_col[t]]
static const char* GRAD_VAL_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void sparse_grad_val(device const float* grad_out    [[buffer(0)]],
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

static inline NSUInteger tgSize(id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger cap = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    NSUInteger tg = MIN(cap, n);
    return tg > 0 ? tg : 1;
}

// ---- forward (CSR) ----
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
    static id<MTLComputePipelineState> pso = makePipeline(device, FWD_GX_SRC, "sparse_forward");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    NSUInteger n = (NSUInteger)B * out_f;
    dispatch_sync(stream->queue(), ^() {
        // torch may hold an open (coalescing) encoder on this command buffer;
        // close it before we start our own, else Metal asserts.
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(x) offset:x.storage_offset()*x.element_size() atIndex:0];
        [enc setBuffer:getMTLBufferStorage(values) offset:values.storage_offset()*values.element_size() atIndex:1];
        [enc setBuffer:getMTLBufferStorage(crow) offset:crow.storage_offset()*crow.element_size() atIndex:2];
        [enc setBuffer:getMTLBufferStorage(csr_col) offset:csr_col.storage_offset()*csr_col.element_size() atIndex:3];
        [enc setBuffer:getMTLBufferStorage(out) offset:0 atIndex:4];
        [enc setBytes:&B length:sizeof(B) atIndex:5];
        [enc setBytes:&in_f length:sizeof(in_f) atIndex:6];
        [enc setBytes:&out_f length:sizeof(out_f) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    });
    return out;
}

// ---- backward grad_val ----
torch::Tensor sparse_grad_val(torch::Tensor grad_out, torch::Tensor x,
                              torch::Tensor csr_rowidx, torch::Tensor csr_col) {
    TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be MPS");
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    uint32_t B = static_cast<uint32_t>(x.size(0));
    uint32_t in_f = static_cast<uint32_t>(x.size(1));
    uint32_t out_f = static_cast<uint32_t>(grad_out.size(1));
    uint32_t K = static_cast<uint32_t>(csr_col.numel());
    auto grad_val = torch::empty({(int64_t)K}, x.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    static id<MTLComputePipelineState> pso = makePipeline(device, GRAD_VAL_SRC, "sparse_grad_val");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    NSUInteger n = (NSUInteger)K;
    dispatch_sync(stream->queue(), ^() {
        // torch may hold an open (coalescing) encoder on this command buffer;
        // close it before we start our own, else Metal asserts.
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(grad_out) offset:grad_out.storage_offset()*grad_out.element_size() atIndex:0];
        [enc setBuffer:getMTLBufferStorage(x) offset:x.storage_offset()*x.element_size() atIndex:1];
        [enc setBuffer:getMTLBufferStorage(csr_rowidx) offset:csr_rowidx.storage_offset()*csr_rowidx.element_size() atIndex:2];
        [enc setBuffer:getMTLBufferStorage(csr_col) offset:csr_col.storage_offset()*csr_col.element_size() atIndex:3];
        [enc setBuffer:getMTLBufferStorage(grad_val) offset:0 atIndex:4];
        [enc setBytes:&B length:sizeof(B) atIndex:5];
        [enc setBytes:&in_f length:sizeof(in_f) atIndex:6];
        [enc setBytes:&out_f length:sizeof(out_f) atIndex:7];
        [enc setBytes:&K length:sizeof(K) atIndex:8];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    });
    return grad_val;
}

// ---- backward grad_x (CSC) ----
torch::Tensor sparse_grad_x(torch::Tensor grad_out, torch::Tensor values,
                            torch::Tensor ccol, torch::Tensor csc_rowidx,
                            torch::Tensor csc_valperm, int64_t in_features) {
    TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be MPS");
    grad_out = grad_out.contiguous();
    values = values.contiguous();
    uint32_t B = static_cast<uint32_t>(grad_out.size(0));
    uint32_t out_f = static_cast<uint32_t>(grad_out.size(1));
    uint32_t in_f = static_cast<uint32_t>(in_features);
    auto grad_x = torch::empty({(int64_t)B, (int64_t)in_f}, grad_out.options());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    static id<MTLComputePipelineState> pso = makePipeline(device, FWD_GX_SRC, "sparse_grad_x");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    NSUInteger n = (NSUInteger)B * in_f;
    dispatch_sync(stream->queue(), ^() {
        // torch may hold an open (coalescing) encoder on this command buffer;
        // close it before we start our own, else Metal asserts.
        stream->endKernelCoalescing();
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(grad_out) offset:grad_out.storage_offset()*grad_out.element_size() atIndex:0];
        [enc setBuffer:getMTLBufferStorage(values) offset:values.storage_offset()*values.element_size() atIndex:1];
        [enc setBuffer:getMTLBufferStorage(ccol) offset:ccol.storage_offset()*ccol.element_size() atIndex:2];
        [enc setBuffer:getMTLBufferStorage(csc_rowidx) offset:csc_rowidx.storage_offset()*csc_rowidx.element_size() atIndex:3];
        [enc setBuffer:getMTLBufferStorage(csc_valperm) offset:csc_valperm.storage_offset()*csc_valperm.element_size() atIndex:4];
        [enc setBuffer:getMTLBufferStorage(grad_x) offset:0 atIndex:5];
        [enc setBytes:&B length:sizeof(B) atIndex:6];
        [enc setBytes:&in_f length:sizeof(in_f) atIndex:7];
        [enc setBytes:&out_f length:sizeof(out_f) atIndex:8];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tgSize(pso, n), 1, 1)];
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    });
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_forward", &sparse_forward, "fused sparse forward (CSR)");
    m.def("sparse_grad_val", &sparse_grad_val, "sparse backward grad_val");
    m.def("sparse_grad_x", &sparse_grad_x, "sparse backward grad_x (CSC)");
}
