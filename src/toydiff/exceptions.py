"""
Specific exceptions known to the use of ToyDiff.
"""


class ToyDiffError(Exception):
    """Base class for for exception in this module"""


class GradientShapeError(ToyDiffError):
    """Exception raised when there are errors related to gradient shape"""

    def __init__(self, message) -> None:
        self.message = message


class NullBackwardFunctionError(ToyDiffError):
    """Exception raised when a call to a non-existing backward function is
    made
    """

    def __init__(self, message) -> None:
        self.message = message


class InPlaceModificationError(ToyDiffError):
    """Exception raised when a call to a non-existing backward function is
    made
    """

    def __init__(self, message) -> None:
        self.message = message
