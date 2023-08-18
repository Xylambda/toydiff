"""
Test all implented operations that are exposed as functions.
"""
import toydiff as tdf
import numpy as np
import torch

from toydiff.testing import generate_input
RTOL = 1e-06


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
