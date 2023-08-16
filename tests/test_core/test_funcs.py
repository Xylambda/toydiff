"""
Test all implented operations that are exposed as functions.
"""
import toydiff as tdf
import numpy as np
import torch

RTOL = 1e-06

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


# -----------------------------------------------------------------------------
# --------------------------- TEST UNARY OPERATIONS ---------------------------
# -----------------------------------------------------------------------------
def test_log():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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
    tensor, tensor_torch = generate_input((5, ))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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
    tensor, tensor_torch = generate_input((5, ))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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
    tensor, tensor_torch = generate_input((5, ))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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
    # test 1d
    tensor, tensor_torch = generate_input((3, ))[0]
    out = tdf.reshape(tensor, (-1, 1))
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

    # -------------------------------------------------------------------------
    # test 3d
    tensor, tensor_torch = generate_input((3, 2, 3))[0]
    out = tdf.reshape(tensor, (-1, 1, 1))
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


def test_exp():
    # -------------------------------------------------------------------------
    # test 1d
    tensor, tensor_torch = generate_input((5, ))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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
    tensor, tensor_torch = generate_input((5, 5))[0]
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


def test_mean():
    pass


def test_std():
    pass


# -----------------------------------------------------------------------------
# -------------------------- TEST BINARY OPERATIONS ---------------------------
# -----------------------------------------------------------------------------
def test_add():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.add(t1, t2)
    out_torch = torch.add(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.add(t1, t2)
    out_torch = torch.add(t1_torch, t2_torch)

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


def test_subtract():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.subtract(t1, t2)
    out_torch = torch.subtract(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.subtract(t1, t2)
    out_torch = torch.subtract(t1_torch, t2_torch)

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


def test_matmul():
    # test 2d
    (t1, t1_torch) = generate_input((5,3))[0]
    (t2, t2_torch) = generate_input((3,6))[0]
    out = tdf.matmul(t1, t2)
    out_torch = torch.matmul(t1_torch, t2_torch)

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

    # TODO: test 3d


def test_multiply():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.multiply(t1, t2)
    out_torch = torch.multiply(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.multiply(t1, t2)
    out_torch = torch.multiply(t1_torch, t2_torch)

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


def test_power():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.power(t1, t2)
    out_torch = torch.pow(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.power(t1, t2)
    out_torch = torch.pow(t1_torch, t2_torch)

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


def test_maximum():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.maximum(t1, t2)
    out_torch = torch.maximum(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.maximum(t1, t2)
    out_torch = torch.maximum(t1_torch, t2_torch)

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


def test_minimum():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.minimum(t1, t2)
    out_torch = torch.minimum(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.minimum(t1, t2)
    out_torch = torch.minimum(t1_torch, t2_torch)

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


def test_divide():
    # test 1d
    (t1, t1_torch), (t2, t2_torch) = generate_input((3,))
    out = tdf.divide(t1, t2)
    out_torch = torch.divide(t1_torch, t2_torch)

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

    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,))[0]
    out = tdf.divide(t1, t2)
    out_torch = torch.divide(t1_torch, t2_torch)

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
