import torch
import numpy as np
import avagrad as ag
from avagrad.testing import generate_input


RTOL = 1e-06


def test_reshape():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((3, ))[0]
    out = ag.reshape(tensor, (-1, 1))
    out_torch = torch.reshape(tensor_torch, (-1, 1))

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        tensor.gradient.numpy(), tensor_torch.grad.numpy(), rtol=RTOL
    )

    # -------------------------------------------------------------------------
    # test 2d
    tensor, tensor_torch = generate_input((3, 2))[0]
    out = ag.reshape(tensor, (2, 3))
    out_torch = torch.reshape(tensor_torch, (2, 3))

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        tensor.gradient.numpy(), tensor_torch.grad.numpy(), rtol=RTOL
    )

    # -------------------------------------------------------------------------
    # test 3d
    tensor, tensor_torch = generate_input((3, 2, 3))[0]
    out = ag.reshape(tensor, (-1, 1, 1))
    out_torch = torch.reshape(tensor_torch, (-1, 1, 1))

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        tensor.gradient.numpy(), tensor_torch.grad.numpy(), rtol=RTOL
    )