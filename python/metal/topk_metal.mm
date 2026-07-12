// Fast approximate top-K selection on MPS, ONE command buffer, no host round-trips.
//
// The rowexact grow bottleneck is picking the ~K largest-|v| cells of a p x D
// strip; torch.topk over ~200K elts is ~400-700 us on MPS. We only need the ~K
// largest (unsorted, approximate K is fine for a grow heuristic), so:
//
//   1. hist_bits : histogram of |v| by magnitude, with NO max reduction -- for
//                  x>=0 the raw float bit pattern is monotonic in value, so
//                  bin = bits(|v|) >> SHIFT is already magnitude-ordered.
//   2. find_tau  : one thread scans the histogram from the top, finds the bin
//                  whose cutoff yields >= K elements, writes the threshold bits.
//   3. compact   : append flat indices with bits(|v|) >= tau via atomic counter.
//
// All three run in ONE command buffer (chained compute encoders, in-order,
// writes visible downstream), a single COMMIT_AND_WAIT, one counter readback.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

using at::native::mps::getMTLBufferStorage;

// SHIFT=18 -> bins of ~0.025% relative value width near 1.0; NB=8192 covers all
// finite float magnitudes (0x7F800000 >> 18 = 8160).
static const char* SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint SHIFT = 18u;    // NB=8192 covers all finite floats; fine bins keep
                              // the threshold-bin narrow (good approx quality)

kernel void hist_bits(device const float* v     [[buffer(0)]],
                      device atomic_uint* hist   [[buffer(1)]],
                      constant uint&      n      [[buffer(2)]],
                      constant uint&      NB     [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint bits = as_type<uint>(fabs(v[gid]));
    uint bin = bits >> SHIFT;
    if (bin >= NB) bin = NB - 1;
    atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
}

kernel void find_tau(device const uint* hist     [[buffer(0)]],
                     device uint*       tau_bits  [[buffer(1)]],
                     constant uint&     NB        [[buffer(2)]],
                     constant uint&     K         [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    uint acc = 0u;
    uint bt = 0u;
    for (int b = (int)NB - 1; b >= 0; --b) {
        acc += hist[b];
        if (acc >= K) { bt = (uint)b; break; }
    }
    tau_bits[0] = bt << SHIFT;
}

kernel void compact_bits(device const float* v        [[buffer(0)]],
                         device const uint*  tau_bits  [[buffer(1)]],
                         device int*         outIdx    [[buffer(2)]],
                         device atomic_uint* counter   [[buffer(3)]],
                         constant uint&      n         [[buffer(4)]],
                         constant uint&      cap       [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint bits = as_type<uint>(fabs(v[gid]));
    if (bits >= tau_bits[0]) {
        uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        if (pos < cap) outIdx[pos] = (int)gid;
    }
}
)METAL";

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, const char* fname) {
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
    TORCH_CHECK(lib, "Metal compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fname]];
    TORCH_CHECK(fn, "no Metal function ", fname);
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(pso, "pipeline failed: ", err ? err.localizedDescription.UTF8String : "unknown");
    return pso;
}

static id<MTLComputePipelineState> psoHist(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, "hist_bits"); return p;
}
static id<MTLComputePipelineState> psoTau(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, "find_tau"); return p;
}
static id<MTLComputePipelineState> psoCompact(id<MTLDevice> d) {
    static id<MTLComputePipelineState> p = makePipeline(d, "compact_bits"); return p;
}

static inline void setBuf(id<MTLComputeCommandEncoder> enc, torch::Tensor t, NSUInteger idx) {
    [enc setBuffer:getMTLBufferStorage(t)
            offset:t.storage_offset() * t.element_size() atIndex:idx];
}

static inline NSUInteger tg(id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger cap = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    NSUInteger t = MIN(cap, n);
    return t > 0 ? t : 1;
}

// One command buffer: hist -> find_tau -> compact. Returns EXACTLY K int32 idx
// (mps). find_tau guarantees >= K cells are >= tau, so cap=K fills every slot,
// and all returned cells are in the top set. No counter readback -> no per-call
// host sync -> pipelines across calls (COMMIT, not COMMIT_AND_WAIT).
torch::Tensor approx_topk(torch::Tensor v, int64_t K) {
    TORCH_CHECK(v.device().is_mps() && v.scalar_type() == torch::kFloat, "v mps float");
    v = v.reshape(-1).contiguous();
    uint32_t n = static_cast<uint32_t>(v.numel());
    uint32_t NB = 8192;    // must match SHIFT=18 in the kernel
    uint32_t Ku = static_cast<uint32_t>(K);
    uint32_t cap = Ku;

    auto i32 = v.options().dtype(torch::kInt32);
    auto hist = torch::zeros({(int64_t)NB}, i32);
    auto tau = torch::zeros({1}, i32);
    auto counter = torch::zeros({1}, i32);
    auto outIdx = torch::empty({(int64_t)Ku}, i32);

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        stream->endKernelCoalescing();
        id<MTLCommandBuffer> cb = stream->commandBuffer();

        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:psoHist(device)];
        setBuf(e1, v, 0); setBuf(e1, hist, 1);
        [e1 setBytes:&n length:sizeof(n) atIndex:2];
        [e1 setBytes:&NB length:sizeof(NB) atIndex:3];
        [e1 dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg(psoHist(device), n), 1, 1)];
        [e1 endEncoding];

        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:psoTau(device)];
        setBuf(e2, hist, 0); setBuf(e2, tau, 1);
        [e2 setBytes:&NB length:sizeof(NB) atIndex:2];
        [e2 setBytes:&Ku length:sizeof(Ku) atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e2 endEncoding];

        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:psoCompact(device)];
        setBuf(e3, v, 0); setBuf(e3, tau, 1); setBuf(e3, outIdx, 2); setBuf(e3, counter, 3);
        [e3 setBytes:&n length:sizeof(n) atIndex:4];
        [e3 setBytes:&cap length:sizeof(cap) atIndex:5];
        [e3 dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg(psoCompact(device), n), 1, 1)];
        [e3 endEncoding];

        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    return outIdx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("approx_topk", &approx_topk, "approximate top-K by |v| (Metal, one command buffer)");
}
