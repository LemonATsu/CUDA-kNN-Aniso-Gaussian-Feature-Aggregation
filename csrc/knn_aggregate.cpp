#include <torch/extension.h>
#include "knn_aggregate.h"


std::vector<torch::Tensor> knn_aggregate_forward(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const int K
) {
    /*
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 1) // TODO: how to aniso?
    */
    CHECK_INPUT(q);
    CHECK_INPUT(p);
    CHECK_INPUT(f);
    CHECK_INPUT(sigma);
    return knn_aggregate_forward_cuda(q, p, f, sigma, K);
}


std::vector<torch::Tensor> knn_aggregate_aniso_forward(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const int K
) {
    CHECK_INPUT(q);
    CHECK_INPUT(p);
    CHECK_INPUT(f);
    CHECK_INPUT(sigma);
    CHECK_INPUT(R);
    return {q, p, f, sigma, R};
}


/*
This macro defines the function that will be called when the Python code imports the extension.
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_aggregate_forward", &knn_aggregate_forward, "KNN aggregate forward");
}