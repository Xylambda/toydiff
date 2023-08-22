"""
Collection of initializers. All initializers modify the passed tensors
in-place.
"""
import numpy as np
from toydiff import Tensor
from typing import Literal


def kaiming(
    tensor: Tensor, dist_type: Literal["uniform", "normal"] = "uniform"
) -> None:
    fan = 1
    gain = 2
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3) * std
    shape = tensor.shape
    tensor.value = np.random.uniform(low=-bound, high=bound, size=shape)