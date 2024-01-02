import torch
import cuda_knn_aggregate
from einops import rearrange
from core.knn_aggregate import KNNAnisotropicAggregate
import pytorch3d.transforms as p3dt

def torch_aggregate(
    q: torch.Tensor,
    p: torch.Tensor,
    f: torch.Tensor,
    sigma: torch.Tensor,
    R: torch.Tensor,
    K: int,
):
    B, Q, F = q.shape
    _, P, _ = p.shape
    diff = q[:, :, None, :, None] - p[:, None, :, :, None]
    R = R[:, None, :]
    #sq_dists = ((R @ diff)[..., 0].pow(2.) * sigma[:, None]).sum(dim=-1)
    sq_dists = ((diff[..., None, :, 0] * R).sum(dim=-1).pow(2) * sigma[:, None]).sum(dim=-1)

    sq_dists, k_idxs = torch.topk(sq_dists, K, dim=-1, largest=False)

    offset = torch.arange(B, device=sq_dists.device)[:, None, None] * P
    idxs = rearrange(k_idxs + offset, 'b q k -> (b q) k')
    sq_dists = rearrange(sq_dists, 'b q k -> (b q) k')

    w_all = torch.exp(-sq_dists[..., None])
    w_out = torch.sum(w_all, dim=1)
    w_out = rearrange(w_out, '(b q) 1 -> b q 1', b=B)

    f = rearrange(f, 'b p f -> (b p) f')[idxs]
    f_out = torch.sum(w_all * f, dim=1)
    f_out = rearrange(f_out, '(b q) f -> b q f', b=B)

    sq_dists = rearrange(sq_dists, '(b q) k -> b q k', b=B)
    f = rearrange(f, '(b p) k f -> b p k f', b=B)
    idxs = rearrange(idxs, '(b q) k -> b q k', b=B)
    return f_out,  w_out, w_all, sq_dists, k_idxs

if __name__ == '__main__':
    import time
    B, Q, F = 5, 32, 61
    P = 16
    K = 7

    T = 50
    N = 100 + T

    cnt_torch = 0.0
    cnt_cuda = 0.0
    for i in range(N):
        q = torch.rand(B, Q, 3).cuda() * 2 - 1
        p = torch.rand(B, P, 3).cuda() * 2 - 1
        f = torch.randn(B, P, F).cuda()
        sigma = 1 / ((torch.randn(B, P, 3)).abs() + 1e-5).cuda()
        R = p3dt.axis_angle_to_matrix(torch.randn(B, P, 3).cuda())

        f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)

        if i < T:
            continue


        sigma.requires_grad = True
        start = time.time()
        #f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)
        f_out, w_out, dists, idxs = KNNAnisotropicAggregate.apply(q, p, f, sigma, R, K)
        cnt_cuda += time.time() - start

        f_out.sum().backward()

        p_torch = p.detach().clone().requires_grad_(True)
        f_torch = f.detach().clone().requires_grad_(True)
        sigma_torch = sigma.detach().clone().requires_grad_(True)
        start = time.time()
        f_torch, w_torch, w_all, dist_torch, idxs_torch = torch_aggregate(q, p_torch, f_torch, sigma_torch, R, K)
        cnt_torch += time.time() - start
        f_torch.sum().backward()

        #assert torch.allclose(f_torch, f_out, atol=1e-6)
        if not (torch.allclose(sigma.grad, sigma_torch.grad, atol=1e-5)):
            print("-0----0---")
            print(f"Difference {idxs} -- {idxs_torch}")

        assert torch.allclose(dist_torch, dists, atol=1e-6)
        assert torch.allclose(w_out.sum(dim=-1, keepdim=True), w_torch, atol=1e-5)

    print(f'torch version : {cnt_torch:.6f}')
    print(f'_cuda version : {cnt_cuda:.6f}')

