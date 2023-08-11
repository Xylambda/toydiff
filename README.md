# Toydiff

`toydiff` is a simple automatic differentiation library that I created to wrap
my head around how autodiff works.

It is build using only NumPy and tested against PyTorch.

The libray is very versatile, and can be used to create and train neural
networks (of course).

## Installation
Normal user:
```bash
git clone https://github.com/Xylambda/toydiff.git
pip install toydiff/.
```

alternatively:
```bash
git clone https://github.com/Xylambda/toydiff.git
pip install toydiff/. -r toydiff/requirements-base.txt
```

Developer:
```bash
git clone https://github.com/Xylambda/toydiff.git
pip install -e toydiff/. -r toydiff/requirements-dev.txt
```

## Tests
To run test, you must install the library as a `developer`.
```bash
cd toydiff/
sh run_tests.sh
```

alternatively:
```bash
cd toydiff/
pytest -v tests/
```

## Usage
The use is almost the same as the one you would expect from PyTorch:

```python
>>> import numpy as np
>>> import toydiff as tdf

>>> # use `track_gradient=True` to allow backward to fill the gradients
>>> a = tdf.Tensor(np.random.rand(3,3), track_gradient=True)
>>> b = tdf.Tensor(np.random.rand(3,3), track_gradient=True)
>>> c = tdf.matmul(a, b)
>>> d = tdf.log(c)
>>> e = tdf.sum(d)
```

Variable `e` is a Tensor that allows to backpropagate:
```python
>>> e
>>> Tensor(-4.30126, dtype=float32, backward_fn=<Sum.backward>)
>>> e.backward()  # can pass a gradient tensor too if needed
>>> a.gradient
Tensor([[1.539179  , 3.2685497 , 0.8082636 ],
       [2.9209654 , 6.9494014 , 1.2896122 ],
       [1.768115  , 3.879517  , 0.83321786]], dtype=float32)
```

The library also tracks intermediate gradients by default, no need to perform
extra steps to do so:

```python
>>> d.gradient
Tensor([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
```

## Custom operations
If, for some reason, the pool of operations `toydiff` provides is not enough,
one can easily create one. Let's use matrix multiplication as example, since it
also involves the use of dunder methods.

We first need to decide which type of operation we want; there are 3 types:
* Unary: f(a) = b, keeps dimensions. e.g. log, exp.

* Reduce: f(a) = b, reduce dimensions. e.g. sum, max.

* Binary: f(a, b) = c. e.g. add, matmul.

A matrix multiplication operation is a binary operation, so we need to extend
the corresponding class to create out own:


```python
import toydiff as tdf
from toydiff.core import BinaryOp
```

Unless specific requirements, we do not need to fill the constructor method,
only the `forward`, `backward` and `__repr__` methods:
* The `forward` method must contain the implementation of the operation,
and the output must be wrapped in a Tensor Class. The parameters `parents`
and `track_gradient` should be filled with the ones provide by the base
class, and the parameter `is_leaf` must be set to True, since the output
tensor is not a leaf node of the graph.

* The `backward` pass implemets the gradient calculation for the operands
involved in the operation. Lastly, wrap the arrays in a Tensor class and call
`_set_gradients`. Any returned value will be ignored.


```python
class MatMul(BinaryOp):
    def forward(self, *args, **kwargs):
        data_a, data_b = self.get_value()
        return Tensor(
            np.matmul(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
        )

    def backward(self, gradient: "Tensor" = None):
        grad_np = gradient.numpy()
        gradient_a = Tensor(grad_np @ self.tensor_b.numpy().T)
        gradient_b = Tensor(self.tensor_a.numpy().T @ grad_np)
        self._set_gradients(gradient_a, gradient_b)

    def __repr__(self):
        return "MatMul(BinaryOp)"
```

To call the operation, it is convenient to use the `OperationRunner` class. We
will wrap everything in a function:

```python
from toydiff.core import OperationRunner
def matmul(tensor_a, tensor_b, *args, **kwargs):
    return OperationRunner(MatMul, tensor_a, tensor_b).run(*args, **kwargs)
```

Now, we can monkey patch the Tensor class to add a matmul operation:
```python
setattr(Tensor, "__matmul__", matmul)
```