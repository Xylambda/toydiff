import torch
import numpy as np
import toydiff as tdf
from toydiff.testing import generate_input


RTOL = 1e-06


def test_mean():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))[0]
    out = tdf.mean(tensor)
    out_torch = torch.mean(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))[0]
    out = tdf.mean(tensor)
    out_torch = torch.mean(tensor_torch)

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
    # test 2d (with axis)
    tensor, tensor_torch = generate_input((5, 5))[0]
    out = tdf.mean(tensor, axis=0)
    out_torch = torch.mean(tensor_torch, dim=0)

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
    # test 2d (with axis)
    tensor, tensor_torch = generate_input((5, 5))[0]
    out = tdf.mean(tensor, axis=1)
    out_torch = torch.mean(tensor_torch, dim=1)

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
    # test 3d (with axis)
    tensor, tensor_torch = generate_input((5, 5, 5))[0]
    out = tdf.mean(tensor, axis=0)
    out_torch = torch.mean(tensor_torch, dim=0)

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
    # test 3d (with axis)
    tensor, tensor_torch = generate_input((5, 5, 5))[0]
    out = tdf.mean(tensor, axis=1)
    out_torch = torch.mean(tensor_torch, dim=1)

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