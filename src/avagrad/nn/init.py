"""
Collection of initializers. All initializers modify the passed tensors
in-place.
"""
import numpy as np

from avagrad import Tensor

__all__ = ["kaiming_uniform"]


def kaiming_uniform(tensor: Tensor, gain: int = 1) -> None:
    # https://github.com/rsokl/MyGrad/blob/master/src/mygrad/nnet/initializers/he_uniform.py
    if tensor.ndim == 1:
        fan = 1
    else:
        fan = tensor.shape[1]

    # TODO: fix this kaiming implementation (actually, I should re-read the paper first)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3) * std
    shape = tensor.shape
    tensor.value = np.random.uniform(low=-bound, high=bound, size=shape)
