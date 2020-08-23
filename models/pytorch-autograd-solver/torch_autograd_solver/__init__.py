"""
TODO rewrite Function with @staticmethod and ctx
"""

__version__ = "0.1.0"

import torch
import torch_autograd_solver_aten as A

class BasicSymeig(torch.autograd.Function):
    """Basic (non-batched, non-generalized) eigensystem solver

    Args:
        input = input matrix

    Returns:
        w = eigenvalues

    TODO:
        replace this function with torch.symeig default in pytorch 1.0

    See_also:
    - theano implementation
    https://github.com/Theano/Theano/blob/rel-1.0.3/theano/tensor/nlinalg.py#L294

    - pytorch implementaiton (forked)
    https://gist.github.com/ncullen93/9acefab137976712de0a51d88b39ffe7
    """
    def __init__(self, upper=True):
        self.upper = upper

    def forward(self, input):
        """
        Returns:
            w (or diag matrix W), V, where

        A V = V W
        -> \sum_j A_{i, j} V_{j, k} = w_k V_{i, k}
        A = V W V^T
        -> A_{i, j} = V_{i, j} w_j V_{j, i}

        W = V^T A V
        -> w_k = V {}

        derivative w.r.t. A_{i, j}

        - \sum_{j'} (A_{i, j'} d V_{j', k}) + V_{j, k} = d w_k V_{i, k} + w_k d V_{i, k}
        - 1 = d (V_{i, j} w_j V_{j, i})

        """
        w, v = torch.symeig(input, eigenvectors=True, upper=self.upper)
        self.save_for_backward(input, w, v)
        return w, v

    def backward(self, grad_w, grad_v):
        x, w, v, = self.saved_tensors
        return A.symeig_backward(grad_w, grad_v, x, w, v, self.upper)


def load_cusolver():
    try:
        import torch_cusolver
        return torch_cusolver
    except ImportError:
        assert False, "need to run `pip install git+https://github.com/ShigekiKarita/pytorch-cusolver`"


class BatchSymeig(torch.autograd.Function):
    def __init__(self, upper=True, tol=1e-7, max_sweeps=100):
        self.upper = upper
        self.tol = tol
        self.max_sweeps = max_sweeps

    def forward(self, input):
        if input.is_cuda:
            cusolver = load_cusolver()
            w, V = cusolver.cusolver_batch_eigh(input, False, ~self.upper, self.tol, self.max_sweeps, True)
        else:
            w, V = A.batch_symeig_forward(input, self.upper)
        self.save_for_backward(input, w, V)
        return w, V

    def backward(self, grad_w, grad_v):
        x, w, v = self.saved_tensors
        if grad_w.is_cuda :
            v=v.transpose(1,2)
            grad_v=grad_v.transpose(1,2)
#        print('x[0]',x[0])
#        print('w[0]',w[0])
#        print('v[0]',v[0])
#        print('grad_w[0]',grad_w[0])
#        print('grad_v[0]',grad_v[0])
        
#        if grad_w.is_cuda :
#            tmp=A.batch_symeig_cuda_backward(grad_w, grad_v, x, w, v, self.upper)
#        else:
        tmp=A.batch_symeig_backward(grad_w, grad_v, x, w, v, self.upper)
        return tmp


class GeneralizedSymeig(torch.autograd.Function):
    def __init__(self, use_jacob=False, tol=1e-7, max_sweeps=100):
        self.upper = True # upper FIXME
        self.tol = tol
        self.max_sweeps = max_sweeps

    def forward(self, a, b):
        """
        Returns:
            w, V, where A V = B V W

        \sum_j A_{i, j} V_{j, k} = \sum_j w_j B_{i, j} V_{j, k}

        derivative w.r.t. A_{i, j}

        \sum_{j' \neq j} A_{i, j'} V_{j', k} + V_{j, k} =

        """
        from scipy.linalg import eigh
        assert a.is_cuda == b.is_cuda
        if a.is_cuda:
            cusolver = load_cusolver()
            w, V = torch_cusolver.cusolver_generalized_eigh(a, False, b, False,
                                                            self.use_jacob,
                                                            self.tol, self.max_sweeps)
        else:
            w, V = eigh(a.detach().numpy(), b.detach().numpy(), lower=not self.upper)
            w = torch.from_numpy(w)
            V = torch.from_numpy(V)
        self.save_for_backward(a, b, w, V)
        return w, V

    def backward(self, grad_w, grad_v):
        """

        """
        a, b, w, v = self.saved_tensors
        return A.generalized_symeig_backward(grad_w, grad_v, a, b, w, v, self.upper)


def symeig(a, b=None, **kwargs):
    """generalized/batched symeig wrapper

    TODO:
    - implement buffer option
    - batch symeig
    - generalized symeig
    - complex (Herm) support
    """
    if a.dim() == 2:
        if b is None:
            return BasicSymeig(**kwargs)(a)
        else:
            return GeneralizedSymeig(**kwargs)(a, b)
    elif a.dim() == 3:
        assert b is None, "batched generalized symeig is not supported. consider batch_inv -> batch_svd"
        return BatchSymeig(**kwargs)(a)
    else:
        raise ValueError("unsupported dim: {}".format(a.dim()))
