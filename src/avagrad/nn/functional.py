"""
Pool of composed non-optimizable (stateless) functions. Each function is
created using the basic operations implemented in core.py
"""
from avagrad.core import Tensor, log, maximum

__all__ = [
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "softmin",
    "tanh",
    "mse_loss",
    "mae_loss",
    "cross_entropy_loss",
]


def relu(tensor: Tensor) -> Tensor:
    return maximum(tensor, 0)


def sigmoid(tensor: Tensor) -> Tensor:
    return tensor.sigmoid()


def softmax(
    tensor: Tensor, axis: int = None, keepdims: bool = False
) -> Tensor:
    _exp = tensor.exp()
    return _exp / _exp.sum(axis=axis, keepdims=keepdims)


def log_softmax(
    tensor: Tensor, axis: int = None, keepdims: bool = False
) -> Tensor:
    return softmax(tensor, axis=axis, keepdims=keepdims).log()


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
    """Mean squared error.

    Can be reduced using a sum function too by passing `reduction="sum"`.

    Parameters
    ----------
    output : avagrad.Tensor
        Predicted tensor.
    target : avagrad.Tensor
        Real tensor.

    Returns
    -------
    avagrad.Tensor
        MSE loss.
    """
    if reduction == "mean":
        return ((output - target) ** 2).mean(axis=axis, keepdims=keepdims)
    elif reduction == "sum":
        return ((output - target) ** 2).sum(axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Unsupported reduction func: '{reduction}'")


def mae_loss(
    output: Tensor,
    target: Tensor,
    axis: int = None,
    keepdims: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Mean absolute error.

    Can be reduced using a sum function too by passing `reduction="sum"`.

    Parameters
    ----------
    output : avagrad.Tensor
        Predicted tensor.
    target : avagrad.Tensor
        Real tensor.

    Returns
    -------
    avagrad.Tensor
        MAE loss.
    """
    if reduction == "mean":
        return ((output - target).abs()).mean(axis=axis, keepdims=keepdims)
    elif reduction == "sum":
        return ((output - target).abs()).sum(axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Unsupported reduction func: '{reduction}'")


def cross_entropy_loss(
    output: Tensor,
    target: Tensor,
    axis: int = None,
    keepdims: bool = False,
    reduction: str = "mean",
) -> Tensor:
    if reduction == "mean":
        return (
            (-target * log(softmax(output)))
            .sum(axis=0)
            .mean(axis=axis, keepdims=keepdims)
        )
    elif reduction == "sum":
        return (
            (-target * log(softmax(output)))
            .sum(axis=0)
            .sum(axis=axis, keepdims=keepdims)
        )
    else:
        raise ValueError(f"Unsupported reduction func: '{reduction}'")
