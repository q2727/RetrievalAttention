#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cfloat>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"

#include "batch_gemm_softmax_arch.h"
#include "batch_gemm_softmax_sm120.h"

using namespace cute;

namespace retroinfer::batch_gemm_softmax {

namespace {

constexpr int kSoftmaxChunk = 256;
constexpr int kSoftmaxThreads = 256;

template <typename T>
CUTLASS_DEVICE float to_float(T value) {
  return static_cast<float>(value);
}

template <typename T>
CUTLASS_DEVICE T from_float(float value) {
  return static_cast<T>(value);
}

CUTLASS_DEVICE float block_reduce_max(float value, float* shared) {
  int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

CUTLASS_DEVICE float block_reduce_sum(float value, float* shared) {
  int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename T>
__global__ void rowwise_softmax_postprocess_kernel(
    const T* gemm_output,
    float* norm,
    float* sum,
    T* softmax_output,
    int rows,
    int columns,
    int partial_blocks) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];

  const T* row_input = gemm_output + static_cast<int64_t>(row) * columns;
  T* row_softmax = softmax_output + static_cast<int64_t>(row) * columns;
  float* row_norm = norm + static_cast<int64_t>(row) * partial_blocks;
  float* row_sum = sum + static_cast<int64_t>(row) * partial_blocks;

  for (int block_idx = 0; block_idx < partial_blocks; ++block_idx) {
    int column = block_idx * kSoftmaxChunk + tid;
    float value = -CUDART_INF_F;
    if (column < columns) {
      value = to_float(row_input[column]);
    }

    float chunk_max = block_reduce_max(value, shared);

    float exp_sum = 0.0f;
    if (column < columns) {
      exp_sum = __expf(value - chunk_max);
    }

    float chunk_sum = block_reduce_sum(exp_sum, shared);
    if (tid == 0) {
      row_norm[block_idx] = chunk_max;
      row_sum[block_idx] = chunk_sum;
    }
    __syncthreads();
  }

  float local_max = -CUDART_INF_F;
  for (int block_idx = tid; block_idx < partial_blocks; block_idx += blockDim.x) {
    local_max = fmaxf(local_max, row_norm[block_idx]);
  }
  float global_max = block_reduce_max(local_max, shared);

  float local_sum = 0.0f;
  for (int block_idx = tid; block_idx < partial_blocks; block_idx += blockDim.x) {
    local_sum += row_sum[block_idx] * __expf(row_norm[block_idx] - global_max);
  }
  float global_sum = block_reduce_sum(local_sum, shared);
  float inv_sum = 1.0f / fmaxf(global_sum, 1e-20f);

  if (tid == 0) {
    row_norm[0] = global_max;
    row_sum[0] = inv_sum;
  }
  __syncthreads();

  for (int block_idx = 0; block_idx < partial_blocks; ++block_idx) {
    int column = block_idx * kSoftmaxChunk + tid;
    if (column < columns) {
      float value = to_float(row_input[column]);
      row_softmax[column] = from_float<T>(__expf(value - global_max) * inv_sum);
    }
  }
}

template <typename T>
struct Sm120GemmOnlyKernel {
  using ElementA = T;
  using ElementB = T;
  using ElementC = T;
  using ElementD = T;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ClusterShapeMNK = Shape<_1, _1, _1>;
  using MmaTileShapeMNK = Shape<_128, _128, _64>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD,
      ElementCompute,
      ElementC,
      ElementCompute>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShapeMNK,
      ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      MmaTileShapeMNK,
      ClusterShapeMNK,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename T>
void run_sm120_gemm_only(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  using Kernel = Sm120GemmOnlyKernel<T>;
  using Gemm = typename Kernel::Gemm;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType problem_shape{m, n, k, batch_count};

  auto stride_A =
      cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, cute::make_shape(m, k, batch_count));
  auto stride_B =
      cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, cute::make_shape(n, k, batch_count));
  auto stride_C =
      cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, cute::make_shape(m, n, batch_count));
  auto stride_D =
      cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, cute::make_shape(m, n, batch_count));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = at::cuda::current_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {reinterpret_cast<T*>(A.data_ptr()), stride_A, reinterpret_cast<T*>(B.data_ptr()), stride_B},
      {{alpha, beta},
       reinterpret_cast<T*>(D.data_ptr()),
       stride_C,
       reinterpret_cast<T*>(D.data_ptr()),
       stride_D},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
  CUTLASS_CHECK(gemm.run(stream));
#else
  (void)A;
  (void)B;
  (void)D;
  (void)batch_count;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  (void)stream;
  TORCH_CHECK(
      false,
      "CUTLASS was not compiled with SM120 support. Use CUDA 12.8+ and CUTLASS 4.4.2+.");
#endif
}

template <typename T>
void batch_gemm_softmax_sm120_impl(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
  TORCH_CHECK(Norm.is_cuda(), "Norm must be a CUDA tensor");
  TORCH_CHECK(Sum.is_cuda(), "Sum must be a CUDA tensor");
  TORCH_CHECK(Softmax.is_cuda(), "Softmax must be a CUDA tensor");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
  TORCH_CHECK(Norm.is_contiguous(), "Norm must be contiguous");
  TORCH_CHECK(Sum.is_contiguous(), "Sum must be contiguous");
  TORCH_CHECK(Softmax.is_contiguous(), "Softmax must be contiguous");

  int rows = batch_count * m;
  int partial_blocks = (n + kSoftmaxChunk - 1) / kSoftmaxChunk;

  TORCH_CHECK(D.numel() >= static_cast<int64_t>(rows) * n, "D buffer is too small for SM120 path");
  TORCH_CHECK(Softmax.numel() >= static_cast<int64_t>(rows) * n, "Softmax buffer is too small for SM120 path");
  TORCH_CHECK(Norm.numel() >= static_cast<int64_t>(rows) * partial_blocks, "Norm buffer is too small for SM120 path");
  TORCH_CHECK(Sum.numel() >= static_cast<int64_t>(rows) * partial_blocks, "Sum buffer is too small for SM120 path");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  run_sm120_gemm_only<T>(A, B, D, batch_count, m, n, k, alpha, beta, stream);

  rowwise_softmax_postprocess_kernel<T><<<rows, kSoftmaxThreads, kSoftmaxThreads * sizeof(float), stream>>>(
      reinterpret_cast<const T*>(D.data_ptr()),
      reinterpret_cast<float*>(Norm.data_ptr()),
      reinterpret_cast<float*>(Sum.data_ptr()),
      reinterpret_cast<T*>(Softmax.data_ptr()),
      rows,
      n,
      partial_blocks);

  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "SM120 rowwise softmax kernel launch failed: ", cudaGetErrorString(status));
}

}  // namespace

void batch_gemm_softmax_sm120(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
  int sm_version = current_device_sm();
  TORCH_CHECK(needs_sm120_port(sm_version), "SM120 path only supports RTX 50-series / SM120 devices");

  if (A.dtype() == torch::kBFloat16) {
    batch_gemm_softmax_sm120_impl<cutlass::bfloat16_t>(
        A, B, D, Norm, Sum, Softmax, batch_count, m, n, k, alpha, beta);
  } else if (A.dtype() == torch::kFloat16) {
    batch_gemm_softmax_sm120_impl<cutlass::half_t>(
        A, B, D, Norm, Sum, Softmax, batch_count, m, n, k, alpha, beta);
  } else {
    TORCH_CHECK(false, "SM120 path only supports Float16 and BFloat16");
  }
}

}  // namespace retroinfer::batch_gemm_softmax
