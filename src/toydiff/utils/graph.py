__all__ = ["topological_sort"]


def topological_sort(last):
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
                for child in node.parents:
                    _topological_sort(child)

            t_sort.append(node)

    _topological_sort(last)
    return t_sort
