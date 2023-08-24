import torch
import numpy as np
import avagrad as tdf
from avagrad.testing import generate_input


RTOL = 1e-06


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