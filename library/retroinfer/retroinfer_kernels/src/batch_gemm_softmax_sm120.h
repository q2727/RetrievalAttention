#pragma once

#include <torch/extension.h>

namespace retroinfer::batch_gemm_softmax {

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
    float alpha = 1.0f,
    float beta = 0.0f);

}  // namespace retroinfer::batch_gemm_softmax
