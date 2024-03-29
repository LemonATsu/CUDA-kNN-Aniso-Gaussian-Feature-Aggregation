import torch
import numpy as np
from . import _C


def knn_aggregate_aniso(
    q: torch.Tensor,
    p: torch.Tensor,
    f: torch.Tensor,
    sigma: torch.Tensor,
    R: torch.Tensor,
    K: int,      
):
    """ k-NN feature aggregate using anisotropic gaussian weights.

    Args:
    ----
        q: [B, Q, 3] query point coordinates.
        p: [B, P, 3] surface point coordinates.
        f: [B, P, F] feature of surface points.
        sigma: [B, P, 3] anisotropic gaussian scales along all three axes.
        R: [B, P, 3, 3] anisotropic gaussian rotations.
        K: int, number of nearest neighbors.
    
    Returns:
    --------
        f_out: [B, Q, F] aggregated feature.
        w_out: [B, Q, K] weights. Note that this is not differentiable.
        dists: [B, Q, K] distances. Note that this is not differentiable.
        idxs: [B, Q, K] k-NN indices. Note that this is not differentiable.
    """
    return KNNAnisotropicAggregate.apply(q, p, f, sigma, R, K)

def knn_aggregate_aniso_lookup(
    q: torch.Tensor,
    p: torch.Tensor,
    f: torch.Tensor,
    sigma: torch.Tensor,
    R: torch.Tensor,
    knn_table: torch.Tensor,
    K: int,      
):
    """ k-NN feature aggregate using anisotropic gaussian weights, but with cached nearest neighbors.

    Args:
    ----
        q: [B, Q, 3] query point coordinates.
        p: [B, P, 3] surface point coordinates.
        f: [B, P, F] feature of surface points.
        sigma: [B, P, 3] anisotropic gaussian scales along all three axes.
        R: [B, P, 3, 3] anisotropic gaussian rotations.
        knn_table: [P, K] pre-computed kNN table using surface point p.
        K: int, number of nearest neighbors.
    
    Returns:
    --------
        f_out: [B, Q, F] aggregated feature.
        w_out: [B, Q, K] weights. Note that this is not differentiable.
        dists: [B, Q, K] distances. Note that this is not differentiable.
        idxs: [B, Q, K] k-NN indices. Note that this is not differentiable.
    """
    return KNNLookupAnisotropicAggregate.apply(q, p, f, sigma, R, knn_table, K)

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

        f_out, w_out, dists, idxs = _C.knn_aggregate_aniso_forward(q, p, f, sigma, R, K)
        ctx.save_for_backward(q, p, f, sigma, R, w_out, idxs)
        return f_out, w_out, dists, idxs
    
    @staticmethod
    def backward(
        ctx, 
        grad_f_out: torch.Tensor, 
        grad_w_out: torch.Tensor, 
        grad_dists: torch.Tensor, 
        grad_idxs: torch.Tensor,
    ):
        """ Backward pass. 
        """
        q, p, f, sigma, R, w_out, idxs = ctx.saved_tensors
        
        # df/dw is simply the sum of the feature vector.
        # FIXME: is there a better naming?
        grad_q, grad_p, grad_f, grad_sigma, grad_R = KNNAnisotropicAggregateBackward.apply(
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
        return grad_q, grad_p, grad_f, grad_sigma, grad_R, None
    

class KNNLookupAnisotropicAggregate(KNNAnisotropicAggregate):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        sigma: torch.Tensor,
        R: torch.Tensor,
        knn_table: torch.Tensor,
        K: int,
    ):
        """ Searching the kNN of q in p.
        Args:
            q: [B, Q, 3] query point coordinates.
            p: [B, P, 3] surface point coordinates.
            f: [B, P, F] feature of surface points.
            sigma: [B, P, 3] anisotropic gaussian scales along all three axes.
            R: [B, P, 3, 3] rotation matrices.
            knn_table: [P, K] pre-computed kNN table.
            K: int, number of nearest neighbors.
        """

        f_out, w_out, dists, idxs = _C.knn_lookup_aggregate_aniso_forward(q, p, f, sigma, R, knn_table, K)
        ctx.save_for_backward(q, p, f, sigma, R, w_out, idxs)
        return f_out, w_out, dists, idxs

    @staticmethod
    def backward(
        ctx, 
        grad_f_out: torch.Tensor, 
        grad_w_out: torch.Tensor, 
        grad_dists: torch.Tensor, 
        grad_idxs: torch.Tensor,
    ):
        """ Backward pass. 
        """
        grads = KNNAnisotropicAggregate.backward(
            ctx, 
            grad_f_out, 
            grad_w_out, 
            grad_dists, 
            grad_idxs,
        )

        return grads + (None,)

class KNNPrecomputedAnisotropicAggregate(KNNAnisotropicAggregate):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        sigma: torch.Tensor,
        R: torch.Tensor,
        knn_idxs: torch.Tensor,
        K: int,
    ):
        """ Searching the kNN of q in p.
        Args:
            q: [B, Q, 3] query point coordinates.
            p: [B, P, 3] surface point coordinates.
            f: [B, P, F] feature of surface points.
            sigma: [B, P, 3] anisotropic gaussian scales along all three axes.
            R: [B, P, 3, 3] rotation matrices.
            knn_table: [P, K] pre-computed kNN table.
            K: int, number of nearest neighbors.
        """

        f_out, w_out, dists = _C.knn_precomputed_aggregate_aniso_forward(q, p, f, sigma, R, knn_idxs, K)
        ctx.save_for_backward(q, p, f, sigma, R, w_out, knn_idxs)
        return f_out, w_out, dists

    @staticmethod
    def backward(
        ctx, 
        grad_f_out: torch.Tensor, 
        grad_w_out: torch.Tensor, 
        grad_dists: torch.Tensor, 
    ):
        """ Backward pass. 
        """

        grads = KNNAnisotropicAggregate.backward(
            ctx, 
            grad_f_out, 
            grad_w_out, 
            grad_dists, 
            None,
        )
        torch.save(grad_f_out, 'grad_f.pt')

        return grads + (None,)


class KNNAnisotropicAggregateBackward(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        grad_f_out: torch.Tensor, 
        grad_w_out: torch.Tensor, 
        q: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        sigma: torch.Tensor,
        R: torch.Tensor,
        w_out: torch.Tensor,
        idxs: torch.Tensor,
    ):
        """ Backward pass. 
        """
        # df/dw is simply the sum of the feature vector.
        # FIXME: is there a better naming?
        grad_q, grad_p, grad_f, grad_sigma, grad_R = _C.knn_aggregate_aniso_backward(
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
        #print(f"1st backward {grad_q.abs().max()} {grad_p.abs().max()} {grad_f.abs().max()} {grad_sigma.abs().max()} {grad_R.abs().max()}")
        ctx.save_for_backward(
            grad_f_out,
            grad_w_out,
            q,
            p,
            f,
            sigma,
            R,
            w_out,
            idxs,
        )
        return grad_q, grad_p, grad_f, grad_sigma, grad_R
    
    @staticmethod
    def backward(
        ctx, 
        grad_grad_q: torch.Tensor,
        grad_grad_p: torch.Tensor, 
        grad_grad_f: torch.Tensor, 
        grad_grad_sigma: torch.Tensor,
        grad_grad_R: torch.Tensor,
    ):
        """ Backward pass only for Eikonal 2nd order backprop!
        """
        #assert grad_grad_p.sum() == 0.0, "Only supports backward via gradient of q!"
        #assert grad_grad_f.sum() == 0.0, "Only supports backward via gradient of q!"
        #assert grad_grad_sigma.sum() == 0.0, "Only supports backward via gradient of q!"

        grad_f_out, grad_w_out, q, p, f, sigma, R, w_out, idxs = ctx.saved_tensors 
        grad_f_out_2nd, grad_q_2nd, grad_p_2nd, grad_f_2nd, grad_sigma_2nd, grad_R_2nd = _C.knn_aggregate_aniso_backward_2nd(
            grad_grad_q.contiguous(),
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

        return grad_f_out_2nd, None, grad_q_2nd, grad_p_2nd, grad_f_2nd, grad_sigma_2nd, grad_R_2nd, None, None

