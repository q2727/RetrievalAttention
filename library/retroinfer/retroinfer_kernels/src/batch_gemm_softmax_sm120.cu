#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/softmax.h>

#include <cuda_runtime.h>
#include <tuple>

#include "cutlass/cutlass.h"

#include "batch_gemm_softmax_arch.h"
#include "batch_gemm_softmax_sm120.h"

namespace retroinfer::batch_gemm_softmax {

namespace {

constexpr int kSoftmaxChunk = 256;

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
  TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same dtype");
  TORCH_CHECK(A.scalar_type() == D.scalar_type(), "A and D must have the same dtype");
  TORCH_CHECK(A.scalar_type() == Softmax.scalar_type(), "A and Softmax must have the same dtype");
  TORCH_CHECK(Norm.scalar_type() == torch::kFloat32, "Norm must be float32");
  TORCH_CHECK(Sum.scalar_type() == torch::kFloat32, "Sum must be float32");
  TORCH_CHECK(A.numel() == static_cast<int64_t>(batch_count) * m * k, "A shape does not match batch_count/m/k");
  TORCH_CHECK(B.numel() == static_cast<int64_t>(batch_count) * n * k, "B shape does not match batch_count/n/k");

  auto a_view = A.view({batch_count, m, k});
  auto b_view = B.view({batch_count, n, k});
  auto d_view = D.view({batch_count, m, n});

  int partial_blocks = (n + kSoftmaxChunk - 1) / kSoftmaxChunk;

  int rows = batch_count * m;

  TORCH_CHECK(D.numel() >= static_cast<int64_t>(rows) * n, "D buffer is too small for SM120 path");
  TORCH_CHECK(Softmax.numel() >= static_cast<int64_t>(rows) * n, "Softmax buffer is too small for SM120 path");
  TORCH_CHECK(Norm.numel() >= static_cast<int64_t>(rows) * partial_blocks, "Norm buffer is too small for SM120 path");
  TORCH_CHECK(Sum.numel() >= static_cast<int64_t>(rows) * partial_blocks, "Sum buffer is too small for SM120 path");

  auto gemm_output = at::bmm(a_view, b_view.transpose(1, 2));
  if (alpha != 1.0f) {
    gemm_output = gemm_output.mul(alpha);
  }
  if (beta != 0.0f) {
    gemm_output = gemm_output + d_view.mul(beta);
  }
  d_view.copy_(gemm_output);

  auto softmax_output = at::softmax(gemm_output, -1);
  Softmax.view({batch_count, m, n}).copy_(softmax_output);

  Norm.zero_();
  Sum.zero_();

  auto norm_view = Norm.view({batch_count, m, partial_blocks});
  auto sum_view = Sum.view({batch_count, m, partial_blocks});
  norm_view.select(-1, 0).copy_(std::get<0>(gemm_output.max(-1, false)).to(torch::kFloat32));
  sum_view.select(-1, 0).fill_(1.0f);
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
