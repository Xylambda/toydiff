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
    model_parameters : iterable
        Parameters to optimize.
    lr : float
        Learning rate.
    """
    def __init__(self, model_parameters, lr: float):
        self.model_parameters = model_parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError("Subclasses must overrige this method")


class GradientDescent(Optimizer):
    """Classic gradient descen algorithm.

    Parameters
    ----------
    model_parameters : iterable
        Parameters to optimize.
    lr : float
        Learning rate.
    """
    def __init__(self, model_parameters, lr=3e-4):
        super().__init__(model_parameters=model_parameters, lr=lr)

    def step(self) -> None:
        for parameter in self.model_parameters:
            parameter -= parameter.gradient * self.lr