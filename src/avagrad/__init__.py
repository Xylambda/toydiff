"""Tensor automatic differentiation and neural networks package. """

# relative subpacackges import
from . import exceptions, nn, random, utils
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from avagrad.core import *

__all__ = ["exceptions", "utils", "nn", "Tensor", "random"]
