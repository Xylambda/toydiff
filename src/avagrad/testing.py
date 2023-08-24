"""
Useful functions for testing
"""
import numpy as np
import torch

import avagrad as tdf


def generate_input(shape, n_tensors=1):
    def create(shape):
        arr = np.random.rand(*shape)
        tensor = tdf.Tensor(arr, track_gradient=True)
        torch_tensor = torch.Tensor(arr)
        torch_tensor.requires_grad = True
        return tensor, torch_tensor

    tensors = []
    for _ in (0, n_tensors):
        tensors.append(create(shape))
    return tensors
