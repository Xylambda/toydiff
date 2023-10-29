"""
Specific exceptions known to the use of AvaGrad.
"""


class AvaGradError(Exception):
    """Base class for for exception in this module"""


class NullBackwardFunctionError(AvaGradError):
    """Exception raised when a call to a non-existing backward function is
    made
    """

    def __init__(self, message) -> None:
        self.message = message


class GradientShapeError(AvaGradError):
    """Exception raised when a gradient tensor shape does not match the shape
    of the tensor it is associated with.
    """

    def __init__(self, message) -> None:
        self.message = message


class InplaceModificationError(AvaGradError):
    """Exception raised when user is trying to modify a tensor whose
    `backward_fn` has already been called.
    """

    def __init__(self, message) -> None:
        self.message = message


class ZeroGradientError(AvaGradError):
    """Exception reaised when is not possible to zero the gradient of a Tensor."""

    def __init__(self, message) -> None:
        self.message = message
