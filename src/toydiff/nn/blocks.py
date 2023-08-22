"""
Pool of optimizable building blocks.
"""
from abc import abstractmethod
from toydiff.core import Tensor, randn, matmul, fma


__all__ = ["Module", "Linear"]


class Module:
    __slots__ = ["_parameters"]
    def __init__(self):
        self._parameters = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_parameter(self, key, value):
        self._parameters[key] = value

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def parameters(self):
        yield from self._parameters.values()

    def named_parameters(self):
        yield from self._parameters.items()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    __slots__ = ["in_features", "out_features", "bias", "weights"]
    def __init__(self, in_features, out_features, bias=False):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # attributes
        self.weights, self.bias = self._initialize_parameters(bias)
        self.register_parameter("weight", self.weights)

        if bias:
            self.register_parameter("bias", self.bias)

    def _initialize_parameters(self, bias: Tensor):
        weights = randn(
            (self.out_features, self.in_features), track_gradient=True
        )
        if bias:
            _bias = randn((self.out_features, ), track_gradient=True)
        else:
            _bias = None
        
        return weights, _bias

    def forward(self, X: Tensor) -> Tensor:
        if self.bias is None:
            return matmul(X, self.weights.T)
        else:
            return fma(X, self.weights.T, self.bias)
