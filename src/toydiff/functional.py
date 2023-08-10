"""
Pool of composed functions. Each function is created using the basic operations
implemented in core.py
"""
from toydiff.core import maximum, Tensor

def conv1d():
    pass


def conv2d():
    pass


def conv3d():
    pass


def relu(tensor: Tensor) -> Tensor:
    return maximum(tensor, Tensor(0))


def sigmoid():
    pass