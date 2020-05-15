""" Small automatic differentiation package for scalars. """

# relative subpacackges import
from . import ops
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
