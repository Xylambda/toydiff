"""
Pool of composed non-optimizable (stateless) functions. Each function is
created using the basic operations implemented in core.py
"""
from toydiff.core import Tensor, maximum


__all__ = ["relu", "sigmoid", "softmax", "softmin", "tanh", "mse_loss"]


def relu(tensor: Tensor) -> Tensor:
    return maximum(tensor, 0)


def sigmoid(tensor: Tensor) -> Tensor:
    return tensor.sigmoid()


def softmax(
    tensor: Tensor, axis: int = None, keepdims: bool = False
) -> Tensor:
    _exp = tensor.exp()
    return _exp / _exp.sum(axis=axis, keepdims=keepdims)


def softmin(
    tensor: Tensor, axis: int = None, keepdims: bool = False
) -> Tensor:
    _exp = (-tensor).exp()
    return _exp / _exp.sum(axis=axis, keepdims=keepdims)


def tanh(tensor: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent function"""
    _exp_pos = tensor.exp()
    _exp_neg = (-tensor).exp()
    return (_exp_pos - _exp_neg) / (_exp_pos + _exp_neg)


# -----------------------------------------------------------------------------
# ------------------------------ LOSS FUNCTIONS -------------------------------
# -----------------------------------------------------------------------------
def mse_loss(
    output: Tensor,
    target: Tensor,
    axis: int = None,
    keepdims: bool = False,
    reduction: str = "mean",
) -> Tensor:

    if reduction == "mean":
        return ((output - target) ** 2).mean(axis=axis, keepdims=keepdims)
    elif reduction == "sum":
        return ((output - target) ** 2).sum(axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Unsupported reduction func: '{reduction}'")

