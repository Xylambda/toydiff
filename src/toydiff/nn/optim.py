"""
Collection of optimizers.
"""
from abc import abstractmethod

from toydiff.nn.blocks import Module

__all__ = ["Optimizer", "SGD"]


class Optimizer:
    """Base class to implement optimizers.

    Subclasses must override 'step' method with the proper algorithm to update
    the parameters.

    Parameters
    ----------
    model : iterable
        Model to optimize.
    lr : float
        Learning rate.
    """

    __slots__ = ["model", "lr"]

    def __init__(self, model: Module, lr: float):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError("Subclasses must override this method")

    def zero_grad(self, criterion="none") -> None:
        for parameter in self.model.parameters():
            parameter.zero_grad(criterion)


class SGD(Optimizer):
    """Stochastic gradient descent algorithm.

    The parameter `beta` is not None by default, which means the algorithm
    applies momentum.

    Parameters
    ----------
    model : iterable
        Model to optimize.
    lr : float
        Learning rate.
    beta : float, optional, default: 0.9
    """

    __slots__ = ["beta"]

    def __init__(self, model, lr: float = 3e-4, beta: float = 0.9):
        super().__init__(model=model, lr=lr)
        self.beta = beta

    def step(self) -> None:
        momentum = 0
        beta = self.beta
        lr = self.lr
        for parameter in self.model.parameters():
            new_dir = parameter.gradient.value * lr
            if beta is not None:
                new_dir = new_dir - beta * momentum

            parameter.value = parameter.value - new_dir
            momentum = new_dir

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, beta={self.beta})"
