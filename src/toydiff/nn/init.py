"""
Collection of initializers. All initializers modify the passed tensors
in-place.
"""
import numpy as np

from toydiff import Tensor


def kaiming_uniform(tensor: Tensor, gain: int) -> None:
    # https://github.com/rsokl/MyGrad/blob/master/src/mygrad/nnet/initializers/he_uniform.py
    fan = 1
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3) * std
    shape = tensor.shape
    tensor.value = np.random.uniform(low=-bound, high=bound, size=shape)
