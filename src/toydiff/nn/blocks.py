"""
Pool of optimizable building blocks.
"""
from abc import abstractmethod
from collections import OrderedDict
from itertools import chain
from typing import Dict, Iterator, Tuple, Optional

from toydiff.core import Tensor, fma, matmul
from toydiff.random import randn

__all__ = ["Module", "Linear"]


class Module:
    _parameters: Dict[str, Optional[Tensor]]
    _training: bool
    def __init__(self):
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_training', True)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_parameter(self, key, value) -> None:
        """Registers a parameter to make it optimizable."""
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        # each param should have a unique identifier
        self._parameters[key] = value

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def parameters(self) -> Iterator[Tensor]:
        for _, param in self.named_parameters():
            yield param

    def _named_simple(self) -> Iterator[Tuple[str, Tensor]]:
        yield from self._parameters.items()

    def _named_complex(self) -> Iterator[Tuple[str, Tensor]]:
        # TODO: probably need to traverse the graph of modules to get them
        # topological sort
        for module in self.__dict__:
            if module not in ["_parameters", "_training"]:
                self._parameters[module] = self.__dict__[module].named_parameters()

        # chain generators
        output = chain()
        for gen in self._parameters.values():
            output = chain(output, gen)

        yield from output

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        # simple module if module does not contain other modules
        if any([isinstance(att, Module) for att in self.__dict__.values()]):
            return self._named_complex()
        else:
            return self._named_simple()

    def zero_grad(self, criterion="none") -> None:
        for parameter in self.parameters():
            parameter.zero_grad(criterion)

    def state_dict(self) -> Dict[str, Tensor]:
        state_dict = OrderedDict()
        for i, (name, param) in enumerate(self.named_parameters()):
            state_dict[f"{name}_{i}"] = param
        return state_dict

    def load_parameters(self, named_params_dict):
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Module"


class Linear(Module):
    __slots__ = ["in_features", "out_features", "bias", "weights"]
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # attributes
        self.weights, self.bias = self._initialize_parameters(bias)
        super().register_parameter("weight", self.weights)

        if bias:
            self.register_parameter("bias", self.bias)

    def _initialize_parameters(self, bias: Tensor):
        weights = randn(
            (self.out_features, self.in_features), track_gradient=True
        )
        if bias:
            _bias = randn((self.out_features,), track_gradient=True)
        else:
            _bias = None

        return weights, _bias

    def forward(self, X: Tensor) -> Tensor:
        if self.bias is None:
            return matmul(X, self.weights.T)
        else:
            return fma(X, self.weights.T, self.bias)

    def __repr__(self) -> str:
        in_ = self.in_features
        ou_ = self.out_features
        return f"Linear(in_features={in_}, out_features={ou_})"
