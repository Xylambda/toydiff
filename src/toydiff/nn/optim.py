"""
Collection of optimizers
"""
from abc import abstractmethod


class Optimizer:
    """Base class to code optimizers.

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
    def __init__(self, model, lr: float):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError("Subclasses must override this method")

    def zero_grad(self):
        for parameter in self.model.parameters():
            parameter.zero_grad()


class GradientDescent(Optimizer):
    """Classic gradient descent algorithm.

    Parameters
    ----------
    model : iterable
        Model to optimize.
    lr : float
        Learning rate.
    """
    def __init__(self, model, lr=3e-4):
        super().__init__(model=model, lr=lr)

    def step(self) -> None:
        for parameter in self.model.parameters():
            parameter.value = parameter.value - parameter.gradient.value * self.lr