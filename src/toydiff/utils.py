"""
Useful utilities for the use of toydiff.
"""
from typing import List

__all__ = ["topological_sort", "draw_graph", "collapse", "gradient_collapse"]


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



def collapse(arr, broadcast_arr, to_collapse):
    """
    Parameters
    ----------
    arr : numpy.ndarray
        Array of bigger dimensions. `broadcast_arr` has been broadcast
        to match the shape of `arr`.
    broadcast_arr : numpy.ndarray
        Broacast array in its original form (non-broadcast).
    to_collapse : numpy.ndarry
        Array to collapse to match `broadcast_arr` array. Its shape is
        the result of the operation between `arr` and `broadcast_arr`.
    """
    target_shape = broadcast_arr.shape
    current_shape = to_collapse.shape

    diff_dims = abs(arr.ndim - broadcast_arr.ndim)
    for _ in range(diff_dims):
        to_collapse = to_collapse.sum(axis=0)

    if broadcast_arr.size == 1:
        to_collapse = to_collapse.sum(keepdims=True)

    # if gradient shape does not match, we further collapse
    if target_shape != to_collapse.shape:
        try:
            to_collapse = to_collapse.reshape(target_shape)
        except:
            axes = []
            for i, (d1, d2) in enumerate(zip(to_collapse.shape, target_shape)):
                if d1 != d2:
                    axes.append(i)
            
            for ax in axes:
                to_collapse = to_collapse.sum(axis=ax)

            to_collapse = to_collapse.reshape(target_shape)
    
    return to_collapse


def gradient_collapse(arr_a, arr_b, grad_a, grad_b):
    if grad_a.shape != arr_a.shape:
        # collapse grad_a
        grad_a = collapse(arr_b, arr_a, grad_a)

    if grad_b.shape != arr_b.shape:
        # collapse grad_b
        grad_b = collapse(arr_a, arr_b, grad_b)
    
    return grad_a, grad_b
