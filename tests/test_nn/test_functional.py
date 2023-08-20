import torch
from toydiff.testing import generate_input
from toydiff.nn.functional import (
    relu, sigmoid, softmax, softmin, tanh, mse_loss
)

import numpy as np


RTOL = 1e-05


def test_relu():
    tensor, tensor_torch = generate_input((5, ))[0]
    out = relu(tensor)
    out_torch = torch.nn.functional.relu(tensor_torch)

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


def test_sigmoid():
    tensor, tensor_torch = generate_input((5, ))[0]
    out = sigmoid(tensor)
    out_torch = torch.nn.functional.sigmoid(tensor_torch)

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


def test_softmax():
    tensor, tensor_torch = generate_input((5, ))[0]
    out = softmax(tensor)
    out_torch = torch.nn.functional.softmax(tensor_torch)

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        tensor.gradient.numpy(), tensor_torch.grad.numpy(), rtol=1e-04
    )


def test_softmin():
    tensor, tensor_torch = generate_input((5, ))[0]
    out = softmin(tensor)
    out_torch = torch.nn.functional.softmin(tensor_torch)

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        tensor.gradient.numpy(), tensor_torch.grad.numpy(), rtol=1e-04
    )


def test_tanh():
    tensor, tensor_torch = generate_input((5, ))[0]
    out = tanh(tensor)
    out_torch = torch.nn.functional.tanh(tensor_torch)

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


def test_mse_loss():
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = mse_loss(t1, t2)
    out_torch = torch.nn.functional.mse_loss(t1_torch, t2_torch)

    # call backward
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    # test forward
    np.testing.assert_allclose(
        out.numpy(), out_torch.detach().numpy(), rtol=RTOL
    )

    # test backward
    np.testing.assert_allclose(
        t1.gradient.numpy(), t1_torch.grad.numpy(), rtol=RTOL
    )

    np.testing.assert_allclose(
        t2.gradient.numpy(), t2_torch.grad.numpy(), rtol=RTOL
    )

