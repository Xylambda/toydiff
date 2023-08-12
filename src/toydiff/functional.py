"""
Pool of composed functions. Each function is created using the basic operations
implemented in core.py
"""
from toydiff.core import Tensor, maximum


def conv1d():
    pass


def conv2d():
    pass


def conv3d():
    pass


def relu(tensor: Tensor) -> Tensor:
    return maximum(tensor, Tensor(0))  # DISC: track_gradient=True ?


def sigmoid():
    pass
