#include <torch/extension.h>
#include "mink.cuh"
#include <iostream>

template <typename scalar_t>
__global__ void knn_aggregate_forward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> q,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> p,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> f,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sigma,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> f_out,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> w_out,
    scalar_t* min_dists,
    int64_t* min_idxs,
    const int K,
    const int F
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;
    int offset = b * q.size(1) * K + n_q * K;

    // create a data structure for holding the min-k
    MinK<scalar_t, int64_t> mink(min_dists + offset, min_idxs + offset, K);

    const int P = p.size(1);
    // iterate through all surface points
    for (int p_idx = 0; p_idx < P; ++ p_idx) {
        // iterate through all dimensions (assume q and p is 3 dimension)
        // TODO: extend to arbitrary dimension?
        scalar_t sq_dist = 0;
        for (int d = 0; d < 3; ++d) {
            const scalar_t diff = q[b][n_q][d] - p[b][p_idx][d];
            sq_dist += diff * diff;
        }
        mink.add(sq_dist, p_idx);
    }
    // sort it so every one is happy, O(K^2) but K is small anyway
    mink.sort();

    for (int k = 0; k < mink.size(); ++k) {
        int64_t k_idx = mink.val(k);
        const scalar_t w = __expf(-mink.key(k) * sigma[b][k_idx][0]);

        // go through the feature
        for (int n_f = 0; n_f < F; ++n_f){
            f_out[b][n_q][n_f] += f[b][k_idx][n_f] * w;
        }
        w_out[b][n_q][0] += w;
        //w_out[b][n_q][k] += w;
        // scalar_t sig = sigma[b][k_idx][0];
        // printf("|||%d-%d %d-th: %.4f \n", b, n_q, k, sig);
    }
    // iterate through the min-k elements to acquire the aggregated features
}


std::vector<torch::Tensor> knn_aggregate_forward_cuda(
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
    const int B = q.size(0);
    const int Q = q.size(1);
    const int F = f.size(2);
    // TODO: revisit to see if other configs run faster 
    const dim3 threads(16, 16); // use a total of 256 threads per-block
    const dim3 blocks((B + threads.x - 1) / threads.x, (Q + threads.y - 1) / threads.y);

    // f.options: set data type and device
    // to specify a particular dtype torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device));
    torch::Tensor f_out = torch::zeros({B, Q, F}, f.options());
    torch::Tensor w_out = torch::zeros({B, Q, 1}, f.options());

    // empty goes faster
    torch::Tensor min_dists = torch::empty({B, Q, K}, q.options());
    torch::Tensor min_idxs = torch::empty({B, Q, K}, torch::dtype(torch::kInt64).device(f.device()));

    // float32 or float64
    // AT_DISPATCH_FLOATING_TYPES_HALF -- float16
    // [&]: captures all variables in the function scope by reference --> [...] {} is lambda function
    // RestrictPtrTratis: pointers will not overlap
    AT_DISPATCH_FLOATING_TYPES(f.type(), "knn_aggregate_forward_cuda_kernel", ([&] {
        knn_aggregate_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            p.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            f.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            sigma.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            f_out.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            w_out.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            min_dists.data_ptr<scalar_t>(),
            min_idxs.data_ptr<int64_t>(),
            K,
            f.size(2)
        );
    }));
    // TODO: whatelse do we need for backward?
    return {f_out, w_out, min_dists, min_idxs};
}