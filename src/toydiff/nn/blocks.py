"""
Pool of optimizable building blocks.
"""
from abc import abstractmethod


__all__ = ["Module", "Linear"]


class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Linear(Module):
    __slots__ = ["in_features", "out_features", "bias"]
    def __init__(self, in_features, out_features, bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, *args, **kwargs):
        pass
