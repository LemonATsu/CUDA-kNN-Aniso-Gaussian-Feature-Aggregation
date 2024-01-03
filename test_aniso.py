import torch
import cuda_knn_aggregate
from einops import rearrange
import numpy as np
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
    sq_dists = sq_dists.float()

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
    B, Q, F = 4, 4096, 64
    P = 6400
    K = 8

    T = 50
    N = 100 + T

    cnt_torch = 0.0
    cnt_cuda = 0.0
    problem_cnt = 0
    for i in range(N):
        q = torch.rand(B, Q, 3).cuda() * 2 - 1
        p = torch.rand(B, P, 3).cuda() * 2 - 1
        f = torch.randn(B, P, F).cuda()
        sigma = 1 / ((torch.randn(B, P, 3)).abs() + 1e-5).cuda()
        R = p3dt.axis_angle_to_matrix(torch.randn(B, P, 3).cuda())

        f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)

        if i < T:
            continue

        p.requires_grad = True
        f.requires_grad = True
        sigma.requires_grad = True
        start = time.time()
        #f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)
        f_out, w_out, dists, idxs = KNNAnisotropicAggregate.apply(q, p, f, sigma, R, K)
        cnt_cuda += time.time() - start

        (w_out.sum() + f_out.sum()).backward()



        p_torch = p.detach().clone().requires_grad_(True)
        f_torch = f.detach().clone().requires_grad_(True)
        sigma_torch = sigma.detach().clone().requires_grad_(True)
        start = time.time()
        f_out_torch, w_torch, w_all, dist_torch, idxs_torch = torch_aggregate(q, p_torch, f_torch, sigma_torch, R, K)
        cnt_torch += time.time() - start
        (w_torch.sum() + f_out_torch.sum()).backward()

        #assert torch.allclose(f_torch, f_out, atol=1e-6)
        idxs_sorted = idxs[..., :-1].sort(dim=-1).values
        idxs_torch_sorted = idxs_torch[..., :-1].sort(dim=-1).values
        """
        if not (torch.allclose(sigma.grad, sigma_torch.grad, atol=1e-5)):
            assert torch.allclose(idxs_sorted.float(), idxs_torch_sorted.float(), atol=1e-5)
            print(f'average sigma difference {(sigma.grad - sigma_torch.grad).abs().mean()}')
        """

        if not (torch.allclose(f.grad, f_torch.grad, atol=1e-5)):
            #assert torch.allclose(idxs_sorted.float(), idxs_torch_sorted.float(), atol=1e-5)
            print(f'average f grad difference {(f.grad - f_torch.grad).abs().mean()}')

        if not (torch.allclose(p.grad, p_torch.grad, atol=1e-4)):
            assert torch.allclose(idxs_sorted.float(), idxs_torch_sorted.float(), atol=1e-5)
            p_grad_difference = (p.grad - p_torch.grad).abs().mean()
            print(f'average p grad difference {p_grad_difference}')
            if p_grad_difference > 0.0001:
                import pdb; pdb.set_trace()
                print
            problem_cnt += 1

        assert torch.allclose(dist_torch, dists, atol=1e-6)
        assert torch.allclose(w_out.sum(dim=-1, keepdim=True), w_torch, atol=1e-5)

    print(f'torch version : {cnt_torch:.6f}')
    print(f'_cuda version : {cnt_cuda:.6f}')
    print(f'problem count: {problem_cnt}/{N-T}')

