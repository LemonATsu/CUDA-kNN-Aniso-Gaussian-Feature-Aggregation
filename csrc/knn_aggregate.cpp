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
        sigma: (B, N_surface_pts, 1)
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
    /*
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 3, 3)
        R: (B, N_surface_pts, 3, 3)
    */
    CHECK_INPUT(q);
    CHECK_INPUT(p);
    CHECK_INPUT(f);
    CHECK_INPUT(sigma);
    CHECK_INPUT(R);
    return knn_aggregate_aniso_forward_cuda(q, p, f, sigma, R, K);
}

std::vector<torch::Tensor> knn_aggregate_aniso_backward(
    const torch::Tensor grad_f_out,
    const torch::Tensor grad_w_out,
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const torch::Tensor w_out,
    const torch::Tensor k_idxs
) {
    CHECK_INPUT(grad_f_out);
    CHECK_INPUT(grad_w_out);
    CHECK_INPUT(q);
    CHECK_INPUT(p);
    CHECK_INPUT(f);
    CHECK_INPUT(sigma);
    CHECK_INPUT(R);
    CHECK_INPUT(w_out);
    CHECK_INPUT(k_idxs);
    return knn_aggregate_aniso_backward_cuda(
        grad_f_out, 
        grad_w_out, 
        q, 
        p, 
        f, 
        sigma, 
        R, 
        w_out,
        k_idxs
    );
}


std::vector<torch::Tensor> knn_aggregate_aniso_backward_2nd(
    const torch::Tensor grad_grad_q,
    const torch::Tensor grad_f_out,
    const torch::Tensor grad_w_out,
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const torch::Tensor w_out,
    const torch::Tensor k_idxs
) {
    /* Second order backward
    */
    CHECK_INPUT(grad_grad_q);
    CHECK_INPUT(grad_f_out);
    CHECK_INPUT(grad_w_out);
    CHECK_INPUT(q);
    CHECK_INPUT(p);
    CHECK_INPUT(f);
    CHECK_INPUT(sigma);
    CHECK_INPUT(R);
    CHECK_INPUT(w_out);
    CHECK_INPUT(k_idxs);

    return knn_aggregate_aniso_backward_2nd_cuda(
        grad_grad_q,
        grad_f_out, 
        grad_w_out, 
        q, 
        p, 
        f, 
        sigma, 
        R, 
        w_out,
        k_idxs
    );
}


/*
This macro defines the function that will be called when the Python code imports the extension.
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_aggregate_forward", &knn_aggregate_forward, "KNN aggregate forward");
    m.def("knn_aggregate_aniso_forward", &knn_aggregate_aniso_forward, "KNN aggregate aniso forward");
    m.def("knn_aggregate_aniso_backward", &knn_aggregate_aniso_backward, "KNN aggregate aniso backward");
    m.def("knn_aggregate_aniso_backward_2nd", &knn_aggregate_aniso_backward_2nd, "KNN aggregate aniso second order backward");
}