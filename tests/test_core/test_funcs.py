"""
Test all implented operations that are exposed as functions.
"""
import toydiff as tdf
import numpy as np
import torch

RTOL = 1e-06

def generate_input(shape):
    arr = np.random.rand(*shape)
    tensor = tdf.Tensor(arr, track_gradient=True)
    torch_tensor = torch.Tensor(arr)
    torch_tensor.requires_grad = True
    return tensor, torch_tensor


# -----------------------------------------------------------------------------
# --------------------------- TEST UNARY OPERATIONS ---------------------------
# -----------------------------------------------------------------------------
def test_log():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))
    out = tdf.log(tensor)
    out_torch = torch.log(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.log(tensor)
    out_torch = torch.log(tensor_torch)

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


def test_negative():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))
    out = tdf.negative(tensor)
    out_torch = torch.negative(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.negative(tensor)
    out_torch = torch.negative(tensor_torch)

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


def test_sin():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))
    out = tdf.sin(tensor)
    out_torch = torch.sin(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.sin(tensor)
    out_torch = torch.sin(tensor_torch)

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


def test_cos():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))
    out = tdf.cos(tensor)
    out_torch = torch.cos(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.cos(tensor)
    out_torch = torch.cos(tensor_torch)

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


def test_reshape():
    # -------------------------------------------------------------------------
    # test 2d
    tensor, tensor_torch = generate_input((3, 2))
    out = tdf.reshape(tensor, (2, 3))
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


def test_exp():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))
    out = tdf.exp(tensor)
    out_torch = torch.exp(tensor_torch)

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
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.exp(tensor)
    out_torch = torch.exp(tensor_torch)

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


def test_transpose():
    # -------------------------------------------------------------------------
    # test 2d
    tensor, tensor_torch = generate_input((5, 5))
    out = tdf.transpose(tensor, (1, 0))
    out_torch = torch.permute(tensor_torch, (1, 0))

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


# -----------------------------------------------------------------------------
# -------------------------- TEST BINARY OPERATIONS ---------------------------
# -----------------------------------------------------------------------------
def test_add():
    pass
