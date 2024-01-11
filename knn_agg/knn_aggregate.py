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
        grad_q, grad_p, grad_f, grad_sigma = KNNAnisotropicAggregateBackward.apply(
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
        return grad_q, grad_p, grad_f, grad_sigma, None, None
    

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

        f_out, w_out, dists, idxs = cuda_knn_aggregate.knn_lookup_aggregate_aniso_forward(q, p, f, sigma, R, knn_table, K)
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
        grad_q, grad_p, grad_f, grad_sigma = cuda_knn_aggregate.knn_aggregate_aniso_backward(
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
        return grad_q, grad_p, grad_f, grad_sigma
    
    @staticmethod
    def backward(
        ctx, 
        grad_grad_q: torch.Tensor,
        grad_grad_p: torch.Tensor, 
        grad_grad_f: torch.Tensor, 
        grad_grad_sigma: torch.Tensor,
    ):
        """ Backward pass only for Eikonal 2nd order backprop!
        """
        #assert grad_grad_p.sum() == 0.0, "Only supports backward via gradient of q!"
        #assert grad_grad_f.sum() == 0.0, "Only supports backward via gradient of q!"
        #assert grad_grad_sigma.sum() == 0.0, "Only supports backward via gradient of q!"
        grad_f_out, grad_w_out, q, p, f, sigma, R, w_out, idxs = ctx.saved_tensors 

        grad_p_2nd, grad_f_2nd, grad_sigma_2nd = cuda_knn_aggregate.knn_aggregate_aniso_backward_2nd(
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

        return None, None, None, grad_p_2nd, grad_f_2nd, grad_sigma_2nd, None, None, None