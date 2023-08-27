""""
Test arbitrary graphs (as many random graphs as possible)
"""
import torch
import numpy as np
import avagrad as ag

RTOL = 1e-06


def test_graph_std():
    """Test graph to compute the standard deviation statistic"""
    arr = np.random.rand(5)
    tensor = ag.Tensor(arr, track_gradient=True)
    t_tensor = torch.Tensor(arr)
    t_tensor.requires_grad = True

    std = ag.power(
        ag.power(tensor - tensor.mean(), 2).sum() / len(tensor), 0.5
    )
    std.backward()

    t_std = torch.pow(
        torch.pow(t_tensor - t_tensor.mean(), 2).sum() / len(t_tensor), 0.5
    )
    t_std.backward()

    np.testing.assert_allclose(std.numpy(), t_std.detach().numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor.gradient.numpy(), t_tensor.grad.numpy(), rtol=RTOL)


def test_graph_a():
    arr = np.random.rand(5, 5)
    tensor_a = ag.Tensor(arr, track_gradient=True)
    tensor_b = ag.Tensor(arr * 5, track_gradient=True)

    t_tensor_a = torch.Tensor(arr)
    t_tensor_a.requires_grad = True
    t_tensor_b = torch.Tensor(arr * 5)
    t_tensor_b.requires_grad = True

    out = ag.log(ag.matmul(tensor_a, tensor_b.T)).mean()
    out_t = torch.log(torch.matmul(t_tensor_a, t_tensor_b.T)).mean()

    out.backward()
    out_t.backward()

    np.testing.assert_allclose(out.numpy(), out_t.detach().numpy(), rtol=RTOL)

    # check backward
    np.testing.assert_allclose(tensor_a.gradient.numpy(), t_tensor_a.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_b.gradient.numpy(), t_tensor_b.grad.numpy(), rtol=RTOL)


def test_graph_b():
    arr = np.random.rand(5, 5)
    tensor_a = ag.Tensor(arr, track_gradient=True)
    tensor_b = ag.Tensor(arr * 5, track_gradient=True)

    t_tensor_a = torch.Tensor(arr)
    t_tensor_a.requires_grad = True
    t_tensor_b = torch.Tensor(arr * 5)
    t_tensor_b.requires_grad = True

    out = ag.exp(ag.matmul(tensor_a, tensor_b)).sum()
    out_t = torch.exp(torch.matmul(t_tensor_a, t_tensor_b)).sum()

    out.backward()
    out_t.backward()

    np.testing.assert_allclose(out.numpy(), out_t.detach().numpy(), rtol=RTOL)

    # check backward
    np.testing.assert_allclose(tensor_a.gradient.numpy(), t_tensor_a.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_b.gradient.numpy(), t_tensor_b.grad.numpy(), rtol=RTOL)


def test_graph_c():
    arr = np.random.rand(5, 5)
    tensor_a = ag.Tensor(arr, track_gradient=True)
    tensor_b = ag.Tensor(arr * 5, track_gradient=True)

    t_tensor_a = torch.Tensor(arr)
    t_tensor_a.requires_grad = True
    t_tensor_b = torch.Tensor(arr * 5)
    t_tensor_b.requires_grad = True

    out = ag.matmul(ag.power(tensor_a, 2), tensor_b / -2).mean()
    out_t = torch.matmul(torch.pow(t_tensor_a, 2), t_tensor_b / -2).mean()

    out.backward()
    out_t.backward()

    np.testing.assert_allclose(out.numpy(), out_t.detach().numpy(), rtol=RTOL)

    # check backward
    np.testing.assert_allclose(tensor_a.gradient.numpy(), t_tensor_a.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_b.gradient.numpy(), t_tensor_b.grad.numpy(), rtol=RTOL)


def test_graph_fma():
    arr = np.random.rand(5, 5)
    tensor_a = ag.Tensor(arr, track_gradient=True)
    tensor_b = ag.Tensor(arr * 5, track_gradient=True)
    tensor_c = ag.Tensor(arr[:, [1]], track_gradient=True)

    t_tensor_a = torch.Tensor(arr.copy())
    t_tensor_a.requires_grad = True
    t_tensor_b = torch.Tensor(arr.copy() * 5)
    t_tensor_b.requires_grad = True
    t_tensor_c = torch.Tensor(arr[:, [1]])
    t_tensor_c.requires_grad = True

    out = ag.matmul(tensor_a, tensor_b) + tensor_c
    out_t = torch.matmul(t_tensor_a, t_tensor_b) + t_tensor_c

    out.backward()
    out_t.backward(torch.ones_like(out_t))

    np.testing.assert_allclose(out.numpy(), out_t.detach().numpy(), rtol=RTOL)

    # check backward
    np.testing.assert_allclose(tensor_a.gradient.numpy(), t_tensor_a.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_b.gradient.numpy(), t_tensor_b.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_c.gradient.numpy(), t_tensor_c.grad.numpy(), rtol=RTOL)


def test_graph_fma_1d():
    arr = np.random.rand(1, 1)
    arr_b = np.random.rand(1,)
    tensor_a = ag.Tensor(arr, track_gradient=True)
    tensor_b = ag.Tensor(arr * 5, track_gradient=True)
    tensor_c = ag.Tensor(arr_b, track_gradient=True)

    t_tensor_a = torch.Tensor(arr.copy())
    t_tensor_a.requires_grad = True
    t_tensor_b = torch.Tensor(arr.copy() * 5)
    t_tensor_b.requires_grad = True
    t_tensor_c = torch.Tensor(arr_b)
    t_tensor_c.requires_grad = True

    out = ag.fma(tensor_a, tensor_b, tensor_c)
    out_t = torch.matmul(t_tensor_a, t_tensor_b) + t_tensor_c

    out.backward()
    out_t.backward(torch.ones_like(out_t))

    np.testing.assert_allclose(out.numpy(), out_t.detach().numpy(), rtol=RTOL)

    # check backward
    np.testing.assert_allclose(tensor_a.gradient.numpy(), t_tensor_a.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_b.gradient.numpy(), t_tensor_b.grad.numpy(), rtol=RTOL)
    np.testing.assert_allclose(tensor_c.gradient.numpy(), t_tensor_c.grad.numpy(), rtol=RTOL)
