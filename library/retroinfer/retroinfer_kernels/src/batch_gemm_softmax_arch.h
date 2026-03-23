#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

namespace retroinfer::batch_gemm_softmax {

struct LegacySm80Config {
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int kStages = 4;
};

struct Sm120PortPlan {
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm120;

  // Blackwell / RTX 50-series candidates from CUTLASS 4.x Blackwell docs.
  // The actual port should use the CUTLASS 3.x/4.x collective-builder path,
  // not the legacy CUTLASS 2.x DefaultGemm path used by this file today.
  using CandidateTile0 = cutlass::gemm::GemmShape<64, 64, 128>;
  using CandidateTile1 = cutlass::gemm::GemmShape<64, 128, 128>;
  using CandidateTile2 = cutlass::gemm::GemmShape<128, 64, 128>;
  using CandidateTile3 = cutlass::gemm::GemmShape<128, 128, 128>;

  static constexpr int kRequiredAlignment = 8;
  static constexpr bool kRequiresCollectiveBuilderRewrite = true;
};

inline int current_device_sm() {
  int device_index = 0;
  cudaError_t status = cudaGetDevice(&device_index);
  if (status != cudaSuccess) {
    return -1;
  }

  cudaDeviceProp props {};
  status = cudaGetDeviceProperties(&props, device_index);
  if (status != cudaSuccess) {
    return -1;
  }

  return props.major * 10 + props.minor;
}

inline bool needs_sm120_port(int sm_version) {
  return sm_version >= cutlass::arch::Sm120::kMinComputeCapability;
}

inline const char* sm120_port_message() {
  return "batch_gemm_softmax uses an SM120 CUTLASS GEMM plus a CUDA rowwise softmax postprocess in "
         "`batch_gemm_softmax_sm120.cu`. The fully fused SM120 epilogue is still pending.";
}

}  // namespace retroinfer::batch_gemm_softmax
