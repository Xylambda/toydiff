"""
Utilities related to gradient tracking, shapes and calculation.
"""


class GradientShaper:
    """Class to ease the calculation of gradients.

    It allows to compute the correct shape depending on the operation
    type and the input tensor or tensors.
    """

    def unary_op(self, tensor):
        pass

    def binary_op(self, tensor_a, tensor_b):
        pass

    def reduce_op(self, tensor):
        pass
