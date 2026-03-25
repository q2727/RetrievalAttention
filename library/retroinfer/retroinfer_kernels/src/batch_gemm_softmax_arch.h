#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
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
  return "batch_gemm_softmax uses ATen/cuBLAS for SM120 dense GEMM plus a CUDA rowwise softmax "
         "postprocess in `batch_gemm_softmax_sm120.cu`. A more native SM120 fused kernel is still pending.";
}

}  // namespace retroinfer::batch_gemm_softmax
