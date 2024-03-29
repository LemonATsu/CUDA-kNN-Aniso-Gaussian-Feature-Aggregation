#include <torch/extension.h>
#include "mink.cuh"


template <typename scalar_t>
__global__ void knn_aggregate_aniso_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_f_out,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_w_out,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> R,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
    const torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> k_idxs,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdq,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdp,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdf,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdsigma,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dLdR,
    const int K,
    const int F
) {
    // each thread handle one query point, and does atomic operation
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;

    for (int n_k = 0; n_k < K; ++n_k) {
        int64_t k_idx = k_idxs[b][n_q][n_k]; // get the n-th ponit index

        // prepare the quantities we need
        scalar_t x = q[b][n_q][0] - p[b][k_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][k_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][k_idx][2];

        // apply rotation to the difference
        scalar_t j = R[b][k_idx][0][0] * x + R[b][k_idx][0][1] * y + R[b][k_idx][0][2] * z;
        scalar_t k = R[b][k_idx][1][0] * x + R[b][k_idx][1][1] * y + R[b][k_idx][1][2] * z;
        scalar_t l = R[b][k_idx][2][0] * x + R[b][k_idx][2][1] * y + R[b][k_idx][2][2] * z;
        scalar_t w = w_out[b][n_q][n_k];

        // compyte dwdp
        scalar_t dwdpx = -2 * w * (
            -j * R[b][k_idx][0][0] * sigma[b][k_idx][0] +
            -k * R[b][k_idx][1][0] * sigma[b][k_idx][1] +
            -l * R[b][k_idx][2][0] * sigma[b][k_idx][2]
        );

        scalar_t dwdpy = -2 * w * (
            -j * R[b][k_idx][0][1] * sigma[b][k_idx][0] +
            -k * R[b][k_idx][1][1] * sigma[b][k_idx][1] +
            -l * R[b][k_idx][2][1] * sigma[b][k_idx][2]
        );

        scalar_t dwdpz = -2 * w * (
            -j * R[b][k_idx][0][2] * sigma[b][k_idx][0] +
            -k * R[b][k_idx][1][2] * sigma[b][k_idx][1] +
            -l * R[b][k_idx][2][2] * sigma[b][k_idx][2]
        );

        scalar_t j_sq = j * j;
        scalar_t k_sq = k * k;
        scalar_t l_sq = l * l;

        scalar_t dwdsigmax = -w * j_sq;
        scalar_t dwdsigmay = -w * k_sq;
        scalar_t dwdsigmaz = -w * l_sq;
        scalar_t w_j_sigmax = -2 * w * j * sigma[b][k_idx][0];
        scalar_t w_k_sigmay = -2 * w * k * sigma[b][k_idx][1];
        scalar_t w_l_sigmaz = -2 * w * l * sigma[b][k_idx][2];
        scalar_t dwdRa = w_j_sigmax * x;
        scalar_t dwdRb = w_j_sigmax * y;
        scalar_t dwdRc = w_j_sigmax * z;
        scalar_t dwdRd = w_k_sigmay * x;
        scalar_t dwdRe = w_k_sigmay * y;
        scalar_t dwdRf = w_k_sigmay * z;
        scalar_t dwdRg = w_l_sigmaz * x;
        scalar_t dwdRh = w_l_sigmaz * y;
        scalar_t dwdRi = w_l_sigmaz * z;

        /////////////////////////////
        //     dLdsigma and dLdf   //
        /////////////////////////////
        scalar_t dL1dfo_dfodw = 0.0;
        scalar_t dLdfo = 0.0;
        for (int n_f = 0; n_f < F; ++n_f) {
            dL1dfo_dfodw += grad_f_out[b][n_q][n_f] * f[b][k_idx][n_f]; 
            atomicAdd(&dLdf[b][k_idx][n_f], grad_f_out[b][n_q][n_f] * w);
        }
        // scalar_t dLdw = dL1dfo_dfodw + grad_w_out[b][n_q][n_k];
         scalar_t dLdw = dL1dfo_dfodw;
        // no racing here?
        dLdq[b][n_q][0] += dLdw * -dwdpx;
        dLdq[b][n_q][1] += dLdw * -dwdpy;
        dLdq[b][n_q][2] += dLdw * -dwdpz;

        // use atomicAdd to avoid race condition
        // dLdq and dLdp is differ by a negative sign
        atomicAdd(&dLdp[b][k_idx][0], dLdw * dwdpx);
        atomicAdd(&dLdp[b][k_idx][1], dLdw * dwdpy);
        atomicAdd(&dLdp[b][k_idx][2], dLdw * dwdpz);


        atomicAdd(&dLdsigma[b][k_idx][0], dLdw * dwdsigmax);
        atomicAdd(&dLdsigma[b][k_idx][1], dLdw * dwdsigmay);
        atomicAdd(&dLdsigma[b][k_idx][2], dLdw * dwdsigmaz);

        atomicAdd(&dLdR[b][k_idx][0][0], dLdw * dwdRa);
        atomicAdd(&dLdR[b][k_idx][0][1], dLdw * dwdRb);
        atomicAdd(&dLdR[b][k_idx][0][2], dLdw * dwdRc);
        atomicAdd(&dLdR[b][k_idx][1][0], dLdw * dwdRd);
        atomicAdd(&dLdR[b][k_idx][1][1], dLdw * dwdRe);
        atomicAdd(&dLdR[b][k_idx][1][2], dLdw * dwdRf);
        atomicAdd(&dLdR[b][k_idx][2][0], dLdw * dwdRg);
        atomicAdd(&dLdR[b][k_idx][2][1], dLdw * dwdRh);
        atomicAdd(&dLdR[b][k_idx][2][2], dLdw * dwdRi);
    }
}


template <typename scalar_t>
__global__ void knn_aggregate_aniso_backward_2nd_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_grad_q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_f_out,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_w_out,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> R,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> w_out,
    const torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> k_idxs,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdgrad_f_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdq,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdp,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdf,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dLdsigma,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dLdR,
    const int K,
    const int F
) {
    // TODO: maybe too many variables here..
    // each thread handle one query point, and does atomic operation
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_q = blockIdx.y * blockDim.y + threadIdx.y;

    // skip extraneous compute
    if (b >= q.size(0) || n_q >= q.size(1)) return;
    scalar_t dLddqx = grad_grad_q[b][n_q][0];
    scalar_t dLddqy = grad_grad_q[b][n_q][1];
    scalar_t dLddqz = grad_grad_q[b][n_q][2];

    for (int n_k = 0; n_k < K; ++n_k) {
        int64_t k_idx = k_idxs[b][n_q][n_k]; // get the n-th ponit index

        // prepare the quantities we need
        scalar_t x = q[b][n_q][0] - p[b][k_idx][0];
        scalar_t y = q[b][n_q][1] - p[b][k_idx][1];
        scalar_t z = q[b][n_q][2] - p[b][k_idx][2];

        // these quantities will be used again
        scalar_t sigma_x = sigma[b][k_idx][0];
        scalar_t sigma_y = sigma[b][k_idx][1];
        scalar_t sigma_z = sigma[b][k_idx][2];
        scalar_t Ra = R[b][k_idx][0][0];
        scalar_t Rb = R[b][k_idx][0][1];
        scalar_t Rc = R[b][k_idx][0][2];
        scalar_t Rd = R[b][k_idx][1][0];
        scalar_t Re = R[b][k_idx][1][1];
        scalar_t Rf = R[b][k_idx][1][2];
        scalar_t Rg = R[b][k_idx][2][0];
        scalar_t Rh = R[b][k_idx][2][1];
        scalar_t Ri = R[b][k_idx][2][2];

        scalar_t ax = Ra * x;
        scalar_t by = Rb * y;
        scalar_t cz = Rc * z;
        scalar_t dx = Rd * x;
        scalar_t ey = Re * y;
        scalar_t fz = Rf * z;
        scalar_t gx = Rg * x;
        scalar_t hy = Rh * y;
        scalar_t iz = Ri * z;
        scalar_t j = ax + by + cz;
        scalar_t k = dx + ey + fz;
        scalar_t l = gx + hy + iz;

        // apply rotation to the difference
        // scalar_t j = R[b][k_idx][0][0] * x + R[b][k_idx][0][1] * y + R[b][k_idx][0][2] * z;
        // scalar_t k = R[b][k_idx][1][0] * x + R[b][k_idx][1][1] * y + R[b][k_idx][1][2] * z;
        // scalar_t l = R[b][k_idx][2][0] * x + R[b][k_idx][2][1] * y + R[b][k_idx][2][2] * z;

        scalar_t sigmax_a = sigma_x * Ra;
        scalar_t sigmax_b = sigma_x * Rb;
        scalar_t sigmax_c = sigma_x * Rc;
        scalar_t sigmay_d = sigma_y * Rd;
        scalar_t sigmay_e = sigma_y * Re;
        scalar_t sigmay_f = sigma_y * Rf;
        scalar_t sigmaz_g = sigma_z * Rg;
        scalar_t sigmaz_h = sigma_z * Rh;
        scalar_t sigmaz_i = sigma_z * Ri;


        scalar_t dotx = (j * sigmax_a + k * sigmay_d + l * sigmaz_g);
        scalar_t doty = (j * sigmax_b + k * sigmay_e + l * sigmaz_h);
        scalar_t dotz = (j * sigmax_c + k * sigmay_f + l * sigmaz_i);

        // these are the same as in 1st order
        // TODO: can be replaced by for loop, but do we want to do that?

        scalar_t w = w_out[b][n_q][n_k];
        scalar_t d_dwdqx_dpx = (2 * dotx * dotx - (sigmax_a * Ra + sigmay_d * Rd + sigmaz_g * Rg));
        scalar_t d_dwdqx_dpy = (2 * doty * dotx - (sigmax_a * Rb + sigmay_d * Re + sigmaz_g * Rh));
        scalar_t d_dwdqx_dpz = (2 * dotz * dotx - (sigmax_a * Rc + sigmay_d * Rf + sigmaz_g * Ri));

        scalar_t d_dwdqy_dpx = (2 * dotx * doty - (sigmax_b * Ra + sigmay_e * Rd + sigmaz_h * Rg));
        scalar_t d_dwdqy_dpy = (2 * doty * doty - (sigmax_b * Rb + sigmay_e * Re + sigmaz_h * Rh));
        scalar_t d_dwdqy_dpz = (2 * dotz * doty - (sigmax_b * Rc + sigmay_e * Rf + sigmaz_h * Ri));

        scalar_t d_dwdqz_dpx = (2 * dotx * dotz - (sigmax_c * Ra + sigmay_f * Rd + sigmaz_i * Rg));
        scalar_t d_dwdqz_dpy = (2 * doty * dotz - (sigmax_c * Rb + sigmay_f * Re + sigmaz_i * Rh));
        scalar_t d_dwdqz_dpz = (2 * dotz * dotz - (sigmax_c * Rc + sigmay_f * Rf + sigmaz_i * Ri));

        scalar_t d_dwdqx_dsigmax = 2 * w * j * (j * dotx - Ra);
        scalar_t d_dwdqy_dsigmax = 2 * w * j * (j * doty - Rb);
        scalar_t d_dwdqz_dsigmax = 2 * w * j * (j * dotz - Rc);

        scalar_t d_dwdqx_dsigmay = 2 * w * k * (k * dotx - Rd);
        scalar_t d_dwdqy_dsigmay = 2 * w * k * (k * doty - Re);
        scalar_t d_dwdqz_dsigmay = 2 * w * k * (k * dotz - Rf);

        scalar_t d_dwdqx_dsigmaz = 2 * w * l * (l * dotx - Rg);
        scalar_t d_dwdqy_dsigmaz = 2 * w * l * (l * doty - Rh);
        scalar_t d_dwdqz_dsigmaz = 2 * w * l * (l * dotz - Ri);

        // these can all be for looped
        scalar_t w_sigma_x = -2 * w * sigma_x;
        scalar_t w_sigma_y = -2 * w * sigma_y;
        scalar_t w_sigma_z = -2 * w * sigma_z;

        scalar_t d_dwdqx_dRa = w_sigma_x * (-2 * j * x * dotx + (2 * ax + by + cz));
        scalar_t d_dwdqy_dRa = w_sigma_x * (-2 * j * x * doty + Rb * x);
        scalar_t d_dwdqz_dRa = w_sigma_x * (-2 * j * x * dotz + Rc * x);

        scalar_t d_dwdqx_dRb = w_sigma_x * (-2 * j * y * dotx + Ra * y);
        scalar_t d_dwdqy_dRb = w_sigma_x * (-2 * j * y * doty + (ax + 2 * by + cz));
        scalar_t d_dwdqz_dRb = w_sigma_x * (-2 * j * y * dotz + Rc * y);

        scalar_t d_dwdqx_dRc = w_sigma_x * (-2 * j * z * dotx + Ra * z);
        scalar_t d_dwdqy_dRc = w_sigma_x * (-2 * j * z * doty + Rb * z);
        scalar_t d_dwdqz_dRc = w_sigma_x * (-2 * j * z * dotz + (ax + by + 2 * cz));

        scalar_t d_dwdqx_dRd = w_sigma_y * (-2 * k * x * dotx + (2 * dx + ey + fz));
        scalar_t d_dwdqy_dRd = w_sigma_y * (-2 * k * x * doty + Re * x);
        scalar_t d_dwdqz_dRd = w_sigma_y * (-2 * k * x * dotz + Rf * x);

        scalar_t d_dwdqx_dRe = w_sigma_y * (-2 * k * y * dotx + Rd * y);
        scalar_t d_dwdqy_dRe = w_sigma_y * (-2 * k * y * doty + (dx + 2 * ey + fz));
        scalar_t d_dwdqz_dRe = w_sigma_y * (-2 * k * y * dotz + Rf * y);

        scalar_t d_dwdqx_dRf = w_sigma_y * (-2 * k * z * dotx + Rd * z);
        scalar_t d_dwdqy_dRf = w_sigma_y * (-2 * k * z * doty + Re * z);
        scalar_t d_dwdqz_dRf = w_sigma_y * (-2 * k * z * dotz + (dx + ey + 2 * fz));

        scalar_t d_dwdqx_dRg = w_sigma_z * (-2 * l * x * dotx + (2 * gx + hy + iz));
        scalar_t d_dwdqy_dRg = w_sigma_z * (-2 * l * x * doty + Rh * x);
        scalar_t d_dwdqz_dRg = w_sigma_z * (-2 * l * x * dotz + Ri * x);

        scalar_t d_dwdqx_dRh = w_sigma_z * (-2 * l * y * dotx + Rg * y);
        scalar_t d_dwdqy_dRh = w_sigma_z * (-2 * l * y * doty + (gx + 2 * hy + iz));
        scalar_t d_dwdqz_dRh = w_sigma_z * (-2 * l * y * dotz + Ri * y);

        scalar_t d_dwdqx_dRi = w_sigma_z * (-2 * l * z * dotx + Rg * z);
        scalar_t d_dwdqy_dRi = w_sigma_z * (-2 * l * z * doty + Rh * z);
        scalar_t d_dwdqz_dRi = w_sigma_z * (-2 * l * z * dotz + (gx + hy + 2 * iz));


        /////////////////////////////
        //     dLdsigma and dLdf   //
        /////////////////////////////
        scalar_t dgrad_f_out = -2 * w * (dLddqx * dotx + dLddqy * doty + dLddqz * dotz);
        scalar_t dL1dfo_dfodw = 0.0; 
        for (int n_f = 0; n_f < F; ++n_f) {
            // same as before, but need to multiply by dLddqx/y/z
            scalar_t grad_fo = grad_f_out[b][n_q][n_f];
            dL1dfo_dfodw += grad_fo * f[b][k_idx][n_f]; 
            // use atomicAdd to avoid race condition
            atomicAdd(&dLdf[b][k_idx][n_f], grad_fo * (dgrad_f_out));
            dLdgrad_f_out[b][n_q][n_f] += dgrad_f_out * f[b][k_idx][n_f];
        }

        // dL1dfo_dfodw += grad_w_out[b][n_q][n_k];
        // dLdq and dLdp differs by only a negative sign
        scalar_t dLdfdqx = dL1dfo_dfodw * dLddqx;
        scalar_t dLdfdqy = dL1dfo_dfodw * dLddqy;
        scalar_t dLdfdqz = dL1dfo_dfodw * dLddqz;
        scalar_t dLdqdqx = 2 * w * (dLdfdqx * d_dwdqx_dpx + dLdfdqy * d_dwdqy_dpx + dLdfdqz * d_dwdqz_dpx);
        scalar_t dLdqdqy = 2 * w * (dLdfdqx * d_dwdqx_dpy + dLdfdqy * d_dwdqy_dpy + dLdfdqz * d_dwdqz_dpy);
        scalar_t dLdqdqz = 2 * w * (dLdfdqx * d_dwdqx_dpz + dLdfdqy * d_dwdqy_dpz + dLdfdqz * d_dwdqz_dpz);
        dLdq[b][n_q][0] += dLdqdqx;
        dLdq[b][n_q][1] += dLdqdqy;
        dLdq[b][n_q][2] += dLdqdqz;

        // differ only by a negative sign
        atomicAdd(&dLdp[b][k_idx][0], -dLdqdqx);
        atomicAdd(&dLdp[b][k_idx][1], -dLdqdqy);
        atomicAdd(&dLdp[b][k_idx][2], -dLdqdqz);

        atomicAdd(&dLdsigma[b][k_idx][0], (dLdfdqx * d_dwdqx_dsigmax + dLdfdqy * d_dwdqy_dsigmax + dLdfdqz * d_dwdqz_dsigmax));
        atomicAdd(&dLdsigma[b][k_idx][1], (dLdfdqx * d_dwdqx_dsigmay + dLdfdqy * d_dwdqy_dsigmay + dLdfdqz * d_dwdqz_dsigmay));
        atomicAdd(&dLdsigma[b][k_idx][2], (dLdfdqx * d_dwdqx_dsigmaz + dLdfdqy * d_dwdqy_dsigmaz + dLdfdqz * d_dwdqz_dsigmaz));

        atomicAdd(&dLdR[b][k_idx][0][0], (dLdfdqx * d_dwdqx_dRa + dLdfdqy * d_dwdqy_dRa + dLdfdqz * d_dwdqz_dRa));
        atomicAdd(&dLdR[b][k_idx][0][1], (dLdfdqx * d_dwdqx_dRb + dLdfdqy * d_dwdqy_dRb + dLdfdqz * d_dwdqz_dRb));
        atomicAdd(&dLdR[b][k_idx][0][2], (dLdfdqx * d_dwdqx_dRc + dLdfdqy * d_dwdqy_dRc + dLdfdqz * d_dwdqz_dRc));

        atomicAdd(&dLdR[b][k_idx][1][0], (dLdfdqx * d_dwdqx_dRd + dLdfdqy * d_dwdqy_dRd + dLdfdqz * d_dwdqz_dRd));
        atomicAdd(&dLdR[b][k_idx][1][1], (dLdfdqx * d_dwdqx_dRe + dLdfdqy * d_dwdqy_dRe + dLdfdqz * d_dwdqz_dRe));
        atomicAdd(&dLdR[b][k_idx][1][2], (dLdfdqx * d_dwdqx_dRf + dLdfdqy * d_dwdqy_dRf + dLdfdqz * d_dwdqz_dRf));

        atomicAdd(&dLdR[b][k_idx][2][0], (dLdfdqx * d_dwdqx_dRg + dLdfdqy * d_dwdqy_dRg + dLdfdqz * d_dwdqz_dRg));
        atomicAdd(&dLdR[b][k_idx][2][1], (dLdfdqx * d_dwdqx_dRh + dLdfdqy * d_dwdqy_dRh + dLdfdqz * d_dwdqz_dRh));
        atomicAdd(&dLdR[b][k_idx][2][2], (dLdfdqx * d_dwdqx_dRi + dLdfdqy * d_dwdqy_dRi + dLdfdqz * d_dwdqz_dRi));
    }
}


std::vector<torch::Tensor> knn_aggregate_aniso_backward_cuda(
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
    /*
        grad_f_out: (B, N_query, C) -- gradient w.r.t aggregated features
        grad_w_out: (B, N_query, K) -- gradient w.r.t kNN weights
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 3, 3) 
        R: (B, N_surface_pts, 3, 3)
     */
    const int B = q.size(0);
    const int Q = q.size(1);
    const int P = p.size(1);
    const int F = f.size(2);
    // TODO: revisit to see if other configs run faster 
    const dim3 threads(16, 16); // use a total of 256 threads per-block
    const dim3 blocks((B + threads.x - 1) / threads.x, (Q + threads.y - 1) / threads.y);

    // we need grad_p, grad_f, and grad_sigma
    torch::Tensor dLdq = torch::zeros({B, Q, 3}, q.options());
    torch::Tensor dLdp = torch::zeros({B, P, 3}, p.options());
    torch::Tensor dLdf = torch::zeros({B, P, F}, f.options());
    torch::Tensor dLdsigma = torch::zeros({B, P, 3}, sigma.options());
    torch::Tensor dLdR = torch::zeros({B, P, 3, 3}, R.options());

    int K = k_idxs.size(2);
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_aggregate_aniso_backward_cuda_kernel", ([&] {
        knn_aggregate_aniso_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            k_idxs.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
            dLdq.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdp.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdf.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdsigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdR.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            K,
            F
        );
    }));
    // TODO: whatelse do we need for backward?
    return {dLdq, dLdp, dLdf, dLdsigma, dLdR};
}


std::vector<torch::Tensor> knn_aggregate_aniso_backward_2nd_cuda(
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
    /*
        grad_grad_q: (B, N_query, 3)
        grad_grad_p: (B, N_surface_pts, 3)
        grad_f_out: (B, N_query, C) -- gradient w.r.t aggregated features
        grad_w_out: (B, N_query, K) -- gradient w.r.t kNN weights
        q: (B, N_query, 3)
        p: (B, N_surface_pts, 3)
        f: (B, N_surface_pts, C)
        sigma: (B, N_surface_pts, 3, 3) 
        R: (B, N_surface_pts, 3, 3)
     */
    const int B = q.size(0);
    const int Q = q.size(1);
    const int P = p.size(1);
    const int F = f.size(2);
    // TODO: revisit to see if other configs run faster 
    const dim3 threads(16, 16); // use a total of 256 threads per-block
    const dim3 blocks((B + threads.x - 1) / threads.x, (Q + threads.y - 1) / threads.y);

    // we need grad_p, grad_f, and grad_sigma again
    torch::Tensor dLdgrad_f_out = torch::zeros({B, Q, F}, f.options());
    torch::Tensor dLdq = torch::zeros({B, Q, 3}, p.options());
    torch::Tensor dLdp = torch::zeros({B, P, 3}, p.options());
    torch::Tensor dLdf = torch::zeros({B, P, F}, f.options());
    torch::Tensor dLdsigma = torch::zeros({B, P, 3}, sigma.options());
    torch::Tensor dLdR = torch::zeros({B, P, 3, 3}, R.options());

    int K = k_idxs.size(2);
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "knn_aggregate_aniso_backward_2nd_cuda_kernel", ([&] {
        knn_aggregate_aniso_backward_2nd_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_grad_q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            w_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            k_idxs.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
            dLdgrad_f_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdq.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdp.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdf.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdsigma.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLdR.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            K,
            F
        );
    }));
    return {dLdgrad_f_out, dLdq, dLdp, dLdf, dLdsigma, dLdR};
}
