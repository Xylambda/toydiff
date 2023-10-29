"""
Pseudo-random numbers generator.

Functions are just wrappers aroung numpy random API.
"""
from typing import Tuple

import numpy as np

from avagrad.core import Tensor

__all__ = ["rand", "randn"]


def rand(shape: Tuple[int], track_gradient: bool = False) -> Tensor:
    """Random values in a given shape.

    Create a tensor of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).

    Parameters
    ----------
    shape : tuple of ints
        Shape of the generated tensor.
    track_gradient : bool, optional, default: False
        If True, the created tensor will be ready to track gradients.

    Returns
    -------
    avagrad.Tensor
        Generated tensor.
    """
    return Tensor(np.random.rand(*shape), track_gradient=track_gradient)


def randn(shape: Tuple[int], track_gradient: bool = False) -> Tensor:
    """Return a sample (or samples) from the "standard normal" distribution.

    If positive int_like arguments are provided, randn generates a tensor of
    shape (d0, d1, ..., dn), filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1. A
    single float randomly sampled from the distribution is returned if no
    argument is provided.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the generated tensor.
    track_gradient : bool, optional, default: False
        If True, the created tensor will be ready to track gradients.

    Returns
    -------
    avagrad.Tensor
        Generated tensor.
    """
    return Tensor(np.random.randn(*shape), track_gradient=track_gradient)
