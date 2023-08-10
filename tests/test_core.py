import torch
import numpy as np
import toydiff as tdf


def test_log():
    torch_t = torch.Tensor([1, 2, 3, 4])
    torch_out = torch.log(torch_t)

    tensor = tdf.Tensor([1, 2, 3, 4])
    out = tdf.log(tensor)

    # forward pass
    np.testing.assert_almost_equal(torch_out.numpy(), out.numpy())


def test_negative():
    pass


def test_sigmoid():
    pass


def test_add():
    pass
