#include <torch/extension.h>
#include "mink.cuh"


template <typename scalar_t>
__global__ void knn_aggregate_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
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
        scalar_t sq_dist = 0;
        for (int d = 0; d < 3; ++d) {
            const scalar_t diff = q[b][n_q][d] - p[b][p_idx][d];
            sq_dist += diff * diff;
        }
        mink.add(sq_dist, p_idx);
    }
    // sort it so everyone is happy, O(K^2) but K is small anyway
    mink.sort();

    // iterate through the min-k elements to acquire the aggregated features
    for (int k = 0; k < mink.size(); ++k) {
        int64_t k_idx = mink.val(k);
        const scalar_t w = __expf(-mink.key(k) * sigma[b][k_idx][0]);

        // go through the feature
        for (int n_f = 0; n_f < F; ++n_f){
            f_out[b][n_q][n_f] += f[b][k_idx][n_f] * w;
        }
        w_out[b][n_q][0] += w;
    }
}


template <typename scalar_t>
__global__ void knn_aggregate_aniso_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> R,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
    scalar_t* min_dists,
    int64_t* min_idxs,
    const int K,
    const int F
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;
    // f_out[b][n_q] += torch::matmul(R[b][n_q], f[b][n_q]);
    int offset = b * q.size(1) * K + n_q * K;

    // create a data structure for holding the min-k
    MinK<scalar_t, int64_t> mink(min_dists + offset, min_idxs + offset, K);

    const int P = p.size(1);
    // iterate through all surface points
    for (int p_idx = 0; p_idx < P; ++p_idx) {

        scalar_t x = q[b][n_q][0] - p[b][p_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][p_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][p_idx][2];
        scalar_t j = R[b][p_idx][0][0] * x + R[b][p_idx][0][1] * y + R[b][p_idx][0][2] * z;
        scalar_t k = R[b][p_idx][1][0] * x + R[b][p_idx][1][1] * y + R[b][p_idx][1][2] * z;
        scalar_t l = R[b][p_idx][2][0] * x + R[b][p_idx][2][1] * y + R[b][p_idx][2][2] * z;
        // compute "scaled" square distance ... not sure if this is the best way to do it.
        scalar_t sq_dist = j * j * sigma[b][p_idx][0] + k * k * sigma[b][p_idx][1] + l * l * sigma[b][p_idx][2];

        mink.add(sq_dist, p_idx);
    }
    // sort it so everyone is happy, O(K^2) but K is small anyway
    mink.sort();

    // iterate through the min-k elements to acquire the aggregated features
    for (int k = 0; k < mink.size(); ++k) {
        int64_t k_idx = mink.val(k);
        const scalar_t w = __expf(-mink.key(k));

        // go through the feature
        for (int n_f = 0; n_f < F; ++n_f){
            f_out[b][n_q][n_f] += f[b][k_idx][n_f] * w;
        }
        w_out[b][n_q][k] = w;
    }
}


template <typename scalar_t>
__global__ void knn_lookup_aggregate_aniso_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> R,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> knn_table,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
    scalar_t* min_dists,
    int64_t* min_idxs,
    const int K,
    const int F
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;
    // f_out[b][n_q] += torch::matmul(R[b][n_q], f[b][n_q]);
    int offset = b * q.size(1) * K + n_q * K;


    const int P = p.size(1);
    scalar_t min_dist = 1e10;
    int64_t min_idx = 0;
    // iterate through all surface points
    for (int p_idx = 0; p_idx < P; ++ p_idx) {

        scalar_t x = q[b][n_q][0] - p[b][p_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][p_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][p_idx][2];
        /*
        // now these are only computed for the tabluated points, should save a bunch of computes?
        scalar_t j = R[b][p_idx][0][0] * x + R[b][p_idx][0][1] * y + R[b][p_idx][0][2] * z;
        scalar_t k = R[b][p_idx][1][0] * x + R[b][p_idx][1][1] * y + R[b][p_idx][1][2] * z;
        scalar_t l = R[b][p_idx][2][0] * x + R[b][p_idx][2][1] * y + R[b][p_idx][2][2] * z;
        // compute "scaled" square distance ... not sure if this is the best way to do it.
        scalar_t sq_dist = j * j * sigma[b][p_idx][0] + k * k * sigma[b][p_idx][1] + l * l * sigma[b][p_idx][2];
        */
        scalar_t sq_dist = x * x + y * y + z * z;
        if (sq_dist < min_dist) {
            min_dist = sq_dist;
            min_idx = p_idx;
        }

    }
    // sort it so everyone is happy, O(K^2) but K is small anyway

    // iterate through the min-k elements to acquire the aggregated features
    for (int n_k = 0; n_k < K; ++n_k) {
        int64_t k_idx = knn_table[min_idx][n_k];
        scalar_t x = q[b][n_q][0] - p[b][k_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][k_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][k_idx][2];
        scalar_t j = R[b][k_idx][0][0] * x + R[b][k_idx][0][1] * y + R[b][k_idx][0][2] * z;
        scalar_t k = R[b][k_idx][1][0] * x + R[b][k_idx][1][1] * y + R[b][k_idx][1][2] * z;
        scalar_t l = R[b][k_idx][2][0] * x + R[b][k_idx][2][1] * y + R[b][k_idx][2][2] * z;
        scalar_t sq_dist = j * j * sigma[b][k_idx][0] + k * k * sigma[b][k_idx][1] + l * l * sigma[b][k_idx][2];
        const scalar_t w = __expf(-sq_dist);

        // go through the feature
        for (int n_f = 0; n_f < F; ++n_f){
            f_out[b][n_q][n_f] += f[b][k_idx][n_f] * w;
        }
        w_out[b][n_q][n_k] = w;
        min_idxs[offset + n_k] = k_idx;
        min_dists[offset + n_k] = sq_dist;
    }
}


template <typename scalar_t>
__global__ void knn_precomputed_aggregate_aniso_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> R,
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> knn_idxs,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
    scalar_t* min_dists,
    const int K,
    const int F
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;
    // f_out[b][n_q] += torch::matmul(R[b][n_q], f[b][n_q]);
    int offset = b * q.size(1) * K + n_q * K;
    const int P = p.size(1);

    // sort it so everyone is happy, O(K^2) but K is small anyway

    // iterate through the min-k elements to acquire the aggregated features
    for (int n_k = 0; n_k < K; ++n_k) {
        int64_t k_idx = knn_idxs[b][n_q][n_k];
        scalar_t x = q[b][n_q][0] - p[b][k_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][k_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][k_idx][2];
        scalar_t j = R[b][k_idx][0][0] * x + R[b][k_idx][0][1] * y + R[b][k_idx][0][2] * z;
        scalar_t k = R[b][k_idx][1][0] * x + R[b][k_idx][1][1] * y + R[b][k_idx][1][2] * z;
        scalar_t l = R[b][k_idx][2][0] * x + R[b][k_idx][2][1] * y + R[b][k_idx][2][2] * z;
        scalar_t sq_dist = j * j * sigma[b][k_idx][0] + k * k * sigma[b][k_idx][1] + l * l * sigma[b][k_idx][2];
        const scalar_t w = __expf(-sq_dist);

        // go through the feature
        for (int n_f = 0; n_f < F; ++n_f){
            f_out[b][n_q][n_f] += f[b][k_idx][n_f] * w;
        }
        w_out[b][n_q][n_k] = w;
        min_dists[offset + n_k] = sq_dist;
    }
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
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_aggregate_forward_cuda_kernel", ([&] {
        knn_aggregate_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            min_dists.data_ptr<scalar_t>(),
            min_idxs.data_ptr<int64_t>(),
            K,
            f.size(2)
        );
    }));
    // TODO: whatelse do we need for backward?
    //return {f_out, f_cache_out, w_out, min_dists, min_idxs};
    return {f_out, w_out, min_dists, min_idxs};
}


std::vector<torch::Tensor> knn_aggregate_aniso_forward_cuda(
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
    const int B = q.size(0);
    const int Q = q.size(1);
    const int F = f.size(2);
    // TODO: revisit to see if other configs run faster 
    const dim3 threads(16, 16); // use a total of 256 threads per-block
    const dim3 blocks((B + threads.x - 1) / threads.x, (Q + threads.y - 1) / threads.y);

    // f.options: set data type and device
    // to specify a particular dtype torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device));
    torch::Tensor f_out = torch::zeros({B, Q, F}, f.options());

    // empty goes faster
    torch::Tensor w_out = torch::empty({B, Q, K}, f.options());
    torch::Tensor min_dists = torch::empty({B, Q, K}, q.options());
    torch::Tensor min_idxs = torch::empty({B, Q, K}, torch::dtype(torch::kInt64).device(f.device()));

    // float32 or float64
    // AT_DISPATCH_FLOATING_TYPES_HALF -- float16
    // [&]: captures all variables in the function scope by reference --> [...] {} is lambda function
    // RestrictPtrTratis: pointers will not overlap
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_aggregate_aniso_forward_cuda_kernel", ([&] {
        knn_aggregate_aniso_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            min_dists.data_ptr<scalar_t>(),
            min_idxs.data_ptr<int64_t>(),
            K,
            f.size(2)
        );
    }));
    // TODO: whatelse do we need for backward?
    return {f_out, w_out, min_dists, min_idxs};
}


std::vector<torch::Tensor> knn_lookup_aggregate_aniso_forward_cuda(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const torch::Tensor knn_table,
    const int K
) {
    /*
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 3, 3) 
        R: (B, N_surface_pts, 3, 3)
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

    // empty goes faster
    torch::Tensor w_out = torch::empty({B, Q, K}, f.options());
    torch::Tensor min_dists = torch::empty({B, Q, K}, q.options());
    torch::Tensor min_idxs = torch::empty({B, Q, K}, torch::dtype(torch::kInt64).device(f.device()));

    // float32 or float64
    // AT_DISPATCH_FLOATING_TYPES_HALF -- float16
    // [&]: captures all variables in the function scope by reference --> [...] {} is lambda function
    // RestrictPtrTratis: pointers will not overlap
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_lookup_aggregate_aniso_forward_cuda_kernel", ([&] {
        knn_lookup_aggregate_aniso_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            knn_table.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            min_dists.data_ptr<scalar_t>(),
            min_idxs.data_ptr<int64_t>(),
            K,
            f.size(2)
        );
    }));
    // TODO: whatelse do we need for backward?
    return {f_out, w_out, min_dists, min_idxs};
}


std::vector<torch::Tensor> knn_precomputed_aggregate_aniso_forward_cuda(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor f,
    const torch::Tensor sigma,
    const torch::Tensor R,
    const torch::Tensor knn_idxs,
    const int K
) {
    /*
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 3, 3) 
        R: (B, N_surface_pts, 3, 3)
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

    // empty goes faster
    torch::Tensor w_out = torch::empty({B, Q, K}, f.options());
    torch::Tensor min_dists = torch::empty({B, Q, K}, q.options());

    // float32 or float64
    // AT_DISPATCH_FLOATING_TYPES_HALF -- float16
    // [&]: captures all variables in the function scope by reference --> [...] {} is lambda function
    // RestrictPtrTratis: pointers will not overlap
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_precomputed_aggregate_aniso_forward_cuda_kernel", ([&] {
        knn_precomputed_aggregate_aniso_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            knn_idxs.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            min_dists.data_ptr<scalar_t>(),
            K,
            f.size(2)
        );
    }));
    // TODO: whatelse do we need for backward?
    return {f_out, w_out, min_dists};
}
