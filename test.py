import torch
import cuda_knn_aggregate
from einops import rearrange


def torch_aggregate(
    q: torch.Tensor,
    p: torch.Tensor,
    f: torch.Tensor,
    sigma: torch.Tensor,
    K: int,
):
    B, Q, F = q.shape
    _, P, _ = p.shape
    sq_dists = torch.sum((q[:, :, None, :] - p[:, None, :, :]) ** 2, dim=-1)
    sq_dists, idxs = torch.topk(sq_dists, K, dim=-1, largest=False)

    sq_dists = rearrange(sq_dists, 'b q k -> (b q) k')
    offset = torch.arange(B, device=sq_dists.device)[:, None, None] * P
    idxs = rearrange(idxs + offset, 'b q k -> (b q) k')
    f = rearrange(f, 'b p f -> (b p) f')[idxs]
    sigma = rearrange(sigma, 'b p 1 -> (b p) 1')[idxs]

    w_all = torch.exp(-sq_dists[..., None] * sigma)
    w_out = torch.sum(w_all, dim=1)
    f_out = torch.sum(w_all * f, dim=1)
    f_out = rearrange(f_out, '(b q) f -> b q f', b=B)
    return f_out, w_out, w_all


if __name__ == '__main__':

    import time
    B, Q, F = 16, 3072, 64
    P = 4096
    K = 24
    N = 100 + 50
    cnt_torch = 0.0

    cnt_cuda = 0.0
    for i in range(N):
        q = torch.randn(B, Q, 3).cuda()
        p = torch.randn(B, P, 3).cuda()
        f = torch.randn(B, P, F).cuda()
        sigma = 1 / (torch.randn(B, P, 1) + 1e-4).abs().cuda()
        start = time.time()
        f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_forward(q, p, f, sigma, K)
        if i < 50:
            continue
        cnt_cuda += time.time() - start

    for i in range(N):
        q = torch.randn(B, Q, 3).cuda()
        p = torch.randn(B, P, 3).cuda()
        f = torch.randn(B, P, F).cuda()
        sigma = 1 / (torch.randn(B, P, 1) + 1e-4).abs().cuda()
        start = time.time()
        f_torch, w_torch, w_all = torch_aggregate(q, p, f, sigma, K)
        if i < 50:
            continue
        cnt_torch += time.time() - start


    print(f'torch version : {cnt_torch:.6f}')
    print(f'_cuda version : {cnt_cuda:.6f}')

    import pdb; pdb.set_trace()
    print