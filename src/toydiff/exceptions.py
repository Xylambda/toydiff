"""
Specific exceptions known to the use of ToyDiff.
"""


class ToyDiffError(Exception):
    """Base class for for exception in this module"""


class NullBackwardFunctionError(ToyDiffError):
    """Exception raised when a call to a non-existing backward function is
    made
    """

    def __init__(self, message) -> None:
        self.message = message


class InplaceModificationError(ToyDiffError):
    """Exception raised when user is trying to modify a tensor whose
    `backward_fn` has already been called.
    """

    def __init__(self, message) -> None:
        self.message = message


class ZeroGradientError(ToyDiffError):
    """Exception reaised when is not possible to zero the gradient of a Tensor.
    """
    def __init__(self, message) -> None:
        self.message = message
