// Minimal MPS custom-op probe: scale a tensor by 2 with a runtime-compiled
// Metal kernel, encoded into PyTorch's own MPS command buffer. Validates the
// whole toolchain (runtime shader compile + getMTLBufferStorage + dispatch)
// before we build the real sparse kernel.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

using at::native::mps::getMTLBufferStorage;

static const char* SCALE_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void scale2(device const float* inp  [[buffer(0)]],
                   device float*       outp [[buffer(1)]],
                   constant uint&      n    [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < n) outp[gid] = inp[gid] * 2.0f;
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

torch::Tensor scale2(torch::Tensor input) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "float32 only");
    input = input.contiguous();
    auto out = torch::empty_like(input);
    uint32_t n = static_cast<uint32_t>(input.numel());

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    static id<MTLComputePipelineState> pso = makePipeline(device, SCALE_SRC, "scale2");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(input)
                offset:input.storage_offset() * input.element_size()
               atIndex:0];
        [enc setBuffer:getMTLBufferStorage(out) offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg > 0 ? tg : 1, 1, 1)];
        [enc endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale2", &scale2, "scale an MPS tensor by 2 (Metal)");
}
