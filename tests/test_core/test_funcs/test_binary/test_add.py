import avagrad as tdf
import numpy as np
import torch

from avagrad.testing import generate_input
RTOL = 1e-06


def test_1d():
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

def test_2d():
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


def test_2d_2d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,1))[0]
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


def test_3d_1d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3,3))[0]
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


def test_3d_2d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3,3))[0]
    (t2, t2_torch) = generate_input((3,1))[0]
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


def test_2d_3d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((3,3))[0]
    (t2, t2_torch) = generate_input((3,1,3))[0]
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


def test_3d_3d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((4,6,2))[0]
    (t2, t2_torch) = generate_input((1,1,1))[0]
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


def test_3d_3d_2():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((4,6,2))[0]
    (t2, t2_torch) = generate_input((1,1,2))[0]
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

def test_4d_1d():
    # -------------------------------------------------------------------------
    # test 2d
    (t1, t1_torch) = generate_input((4,6,2,7))[0]
    (t2, t2_torch) = generate_input((1,))[0]
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
