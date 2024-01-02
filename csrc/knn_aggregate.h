#include <torch/extension.h>
/*
This is a header file for the C++ file.
*/

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> knn_aggregate_forward_cuda(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const int K
);


std::vector<torch::Tensor> knn_aggregate_aniso_forward_cuda(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const int K
);