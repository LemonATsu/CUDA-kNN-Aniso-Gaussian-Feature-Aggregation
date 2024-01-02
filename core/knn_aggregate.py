import torch
import numpy as np
import cuda_knn_aggregate

class KNNAnisotropicAggregate(torch.autograd.Function):
    """ Combining knn search and aggregation into one function.
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        sigma: torch.Tensor,
        R: torch.Tensor,
        K: int,
    ):
        """ Searching the kNN of q in p.
        Args:
            q: [B, Q, 3] query point coordinates.
            p: [B, P, 3] surface point coordinates.
            f: [B, P, F] feature of surface points.
            sigma: [B, P, 3] anisotropic gaussian scales along all three axes.
            R: [B, P, 3, 3] rotation matrices.
            K: int, number of nearest neighbors.
        """

        f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)
        ctx.save_for_backward(q, p, f, sigma, R, w_out, idxs)
        return f_out, w_out, dists, idxs
    
    @staticmethod
    def backward(ctx, grad_f_out, grad_w_out, grad_dists, grad_idxs):
        """ Backward pass. 
        # TODO: grad_w is probably gonna be used for weighting the PE. Handle that!
        """
        q, p, f, sigma, R, w_out, idxs = ctx.saved_tensors
        
        # df/dw is simply the sum of the feature vector.
        # FIXME: is there a better naming?
        grad_p, grad_f, grad_sigma = cuda_knn_aggregate.knn_aggregate_aniso_backward(
            grad_f_out.contiguous(),
            grad_w_out.contiguous(),
            q, 
            p, 
            f, 
            sigma, 
            R, 
            w_out, 
            idxs
        )
        return None, grad_p, grad_f, grad_sigma, None, None