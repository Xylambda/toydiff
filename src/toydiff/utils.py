"""
Useful utilities for the use of toydiff.
"""
from typing import List

__all__ = ["topological_sort"]


def topological_sort(last: "Tensor") -> List["Tensor"]:
    """Topological sort of a graph.

    Assumes `last` is a Tensor with accesible parents to traverse through.

    Parameters
    ----------
    last : Tensor

    Returns
    -------
    t_sort : list of Tensor
        List containing the nodes sorted in topological order.
    """
    t_sort = []
    visited = set()

    def _topological_sort(node):
        if node not in visited:
            visited.add(node)

            if node.parents is not None:
                for parent in node.parents:
                    _topological_sort(parent)

            t_sort.append(node)

    _topological_sort(last)
    return t_sort


def draw_graph(last_node: "Tensor"):
    sorted_tensors = topological_sort(last_node)


class GradientCollapser:
    """Class designed to generate gradient with appropiate shape for a given
    operation.
    """

    # http://coldattic.info/post/116/
    # https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/python/ops/math_grad.py#L63-L84
    def __init__(self,):
        pass

    def binary_grad(self, t1, t2, gradient):
        """
        Parameters
        ----------
        t1 : numpy.ndarray
        t2 : numpy.ndarray
        gradient : numpy.ndarray
            Incoming gradient of the operation.
        """
        diff_dims = t1.ndim - t2.ndim
        for _ in range(diff_dims):
            gradient = gradient.sum(axis=0)

        if t2.size == 1:
            gradient = gradient.sum(keepdims=True)

        return gradient

    def unary_grad(self):
        pass
