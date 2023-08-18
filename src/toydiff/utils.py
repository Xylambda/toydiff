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
