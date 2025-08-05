import torch
from torch import Tensor
from torch.distributions import constraints
from zuko.transforms import Transform

from zuko.distributions import DiagNormal
from zuko.flows import UnconditionalDistribution


class AffineSigmoid(Transform):
    r"""
    Affine Sigmoid transformation that maps from real numbers to the interval (a, b).

    Arguments:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
    """

    bijective = True

    def __init__(self, a: float, b: float, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.domain = constraints.real
        self.codomain = constraints.interval(a, b)

    def _call(self, x: Tensor) -> Tensor:
        # Forward transformation: x in R -> y in (a, b)
        s = torch.sigmoid(x)
        y = self.a + (self.b - self.a) * s
        return y

    def _inverse(self, y: Tensor) -> Tensor:
        # Inverse transformation: y in (a, b) -> x in R
        s = (y - self.a) / (self.b - self.a)
        x = torch.log(s) - torch.log1p(-s)
        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # Log absolute determinant of the Jacobian
        sigma_x = torch.sigmoid(x)
        log_det = torch.log(self.b - self.a) + torch.log(sigma_x) + torch.log1p(-sigma_x)
        return log_det
    
    

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def log_abs_det_jacobian_sigmoid(x,):
    sigma = sigmoid(x)
    return torch.abs(torch.log(sigma) + torch.log(1 - sigma))


