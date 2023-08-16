"""
Pool of composed non-optimizable functions. Each function is created using the
basic operations implemented in core.py
"""
from toydiff.core import Tensor, maximum


__all__ = ["relu", "sigmoid"]


def relu(tensor: Tensor) -> Tensor:
    return maximum(tensor, 0)


def sigmoid():
    pass
