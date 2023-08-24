"""
Useful utilities for the use of toydiff.
"""
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["topological_sort", "draw_graph", "gradient_collapse"]


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
    raise NotImplementedError("Not implement func")


def gradient_collapse(
    data_a: ArrayLike,
    data_b: ArrayLike,
    gradient_a: ArrayLike,
    gradient_b: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Collapse gradient helper function.

    When forward operation needed a broadcast of one of the arrays, the
    gradient of the broadcast array needs to be adjusted to match the original
    array shape.

    See References to know the author of this code.

    Parameters
    ---------
    data_a : numpy.ndarray
    data_b : numpy.ndarray
    gradient : numpy.ndarray
        Incoming gradient of the operations.

    Returns
    -------
    grad_a : numpy.ndarray
    grad_b : numpy.ndarray

    References
    ----------
    .. [1] StackOverflow - More Pythonic Way to Compute derivatives of
       broadcasting addition in numpy
       https://stackoverflow.com/questions/45428696/more-pythonic-way-to-
       compute-derivatives-of-broadcast-addition-in-numpy
    """
    br_a, br_b = np.broadcast_arrays(data_a, data_b)
    axes_a = tuple(
        i
        for i, (da, db) in enumerate(zip(br_a.strides, br_b.strides))
        if da == 0 and db != 0
    )
    axes_b = tuple(
        i
        for i, (da, db) in enumerate(zip(br_a.strides, br_b.strides))
        if da != 0 and db == 0
    )

    collapsed_a = np.sum(gradient_a, axes_a).reshape(data_a.shape)
    collapsed_b = np.sum(gradient_b, axes_b).reshape(data_b.shape)

    return collapsed_a, collapsed_b
