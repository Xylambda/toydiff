"""
Core of the library toydiff. It contains:
    1. A set of composable differentiable operations for toydiff.Tensor
    objects.
    2. A Tensor class.

The module is structured as follows:
    1. First, the base classes are defined. 3 types of operations can be
    performed:
        * Unary: f(a) = b, keeps dimensions. e.g. log, exp.
        * Reduce: f(a) = b, reduce dimensions. e.g. sum, max.
        * Binary: f(a, b) = c. e.g. add, matmul.

    2. Then, a class is defined for each operation. Each class extends the
    appropiate base class.

    3. After each class, a function is created. The function makes use of the
    class and adds the backward function if needed to the result tensor.

    4. The Tensor class is created using the above function and, if possible,
    dunder/magic methods are used to ensure a smooth usage of the library.

"""
import warnings

from abc import abstractmethod
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np

from scipy.special import expit

from toydiff.exceptions import (
    InplaceModificationError,
    NullBackwardFunctionError,
    ZeroGradientError,
    GradientShapeError,
)
from toydiff.utils import topological_sort

__UNARY_OPS = [
    "log",
    "negative",
    "sigmoid",
    "sin",
    "cos",
    "tan",
    "reshape",
    "exp",
    "transpose",
    "sign",
    "abs",
]

__BINARY_OPS = [
    "add",
    "subtract",
    "matmul",
    "multiply",
    "power",
    "maximum",
    "minimum",
    "divide",
]

__REDUCE_OPS = ["max", "min", "sum", "mean", "std"]

# other functions of this module
__OTHER = [
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "empty",
    "empty_like",
    "rand",
    "randn",
    "fma",  # TODO: at some point, we may need to add ternary ops
]

__all__ = ["Tensor"] + __UNARY_OPS + __BINARY_OPS + __REDUCE_OPS + __OTHER


class Operation:
    """Base class to create differentiable operations.

    To create an operation you must implement the forward and backward passes,
    but the forward computation must be calculated calling __call__ and not
    forward.

    Attributes
    ----------
    out : toydiff.Tensor
    """

    __slots__ = ["out", "track_gradient"]

    def __init__(self, track_gradient: bool = False):
        self.track_gradient: bool = track_gradient

        # filled in the forward pass
        self.out: Tensor = None

    def __call__(self, *args, **kwargs) -> "Tensor":
        out = self.forward(*args, **kwargs)
        self.out = out
        return out

    def check_dtype_and_cast(self, obj: object, cast: bool = True) -> None:
        if not isinstance(obj, Tensor):
            if cast:
                return Tensor(obj, is_leaf=True, track_gradient=True)
            else:
                msg = "Operations are supported only for toydiff.Tensor instances"
                raise TypeError(msg)
        else:
            return obj

    def get_gradient(self) -> "Tensor":
        """
        Return the gradient of the forward computation output tensor.
        """
        if self.out is None:
            raise ValueError("Yoy must generate the output tensor")

        return self.out.gradient

    @abstractmethod
    def get_value(self) -> np.ndarray:
        """Retrieves the internal numpy array value of the tensors that compose
        this operation.

        For example, a binary operation will return 2 numpy arrays while a
        unary operation will return one.
        """
        raise NotImplementedError("Subclasses must override this method")

    @abstractmethod
    def forward(self, *args, **kwargs) -> "Tensor":
        """Forward computation operation.

        The operation is performed given the input tensors to produce and
        output result of type Tensor too.

        Although this function implements the computation step, it should not
        be called directly; instead, the __call__ method should be use.

        Returns
        -------
        toydiff.Tensor
            Output tensor.
        """
        raise NotImplementedError("Subclasses must override this method")

    @abstractmethod
    def _set_gradients(self, *args, **kwargs) -> None:
        """Helper method to set the gradients of the parent tensors."""
        raise NotImplementedError("Subclasses must override this method")

    def _backward_fn(self, gradient: Optional["Tensor"] = None) -> None:
        """Actual backward call.

        This method ensures the passed gradient is not None and then calls
        the backward method that is implement for this operation using the
        aforementioned gradient.

        Parameters
        ----------
        gradient : toydiff.Tensor
        """
        if gradient is None:
            gradient = self.get_gradient()

        self.backward(gradient=gradient)

    @abstractmethod
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Backward pass.

        Sets the gradient of this operation with respect to its operands times
        the incoming gradient.

        Parameters
        ----------
        gradient : toydiff.Tensor, optional, default: None
            If None, a Tensor of 1's of the same shape as the output tensor of
            this operation will be used.
        """
        raise NotImplementedError("Subclasses must override this method")


class UnaryOp(Operation):
    """Base class to implement unary operations."""

    __slots__ = ["tensor", "parents", "out"]

    def __init__(self, tensor: "Tensor"):
        tensor = self.check_dtype_and_cast(tensor)

        track_gradient = tensor.track_gradient
        super().__init__(track_gradient=track_gradient)

        self.tensor = tensor
        self.parents = [tensor]

    def get_value(self) -> np.ndarray:
        return self.tensor.numpy()

    def _set_gradients(self, gradient) -> None:
        if self.tensor.track_gradient:
            if self.tensor.shape != gradient.shape:
                msg = (
                    f"Wrong gradient shape {gradient.shape} for a tensor of"
                    f" shape {self.tensor.shape}"
                )
                raise GradientShapeError(msg)
            
            if self.tensor.gradient is None:
                self.tensor.gradient = gradient
            else:
                self.tensor.gradient.value += gradient.value


class BinaryOp(Operation):
    """Base class to implement binary operations.

    The method `get_value` will return the NumPy arrays of the tensor_a and
    tensor_b in the same order they were passed.

    Similarly, the method `_set_gradients` expects the first and second
    arguments to be the gradients for the first and second tensors passed in
    the constructors (respectively).

    Parameters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Attributes
    ----------
    parents : list of toydiff.Tensor
    """

    __slots__ = ["tensor_a", "tensor_b", "parents"]

    def __init__(self, tensor_a: "Tensor", tensor_b: "Tensor"):
        tensor_a = self.check_dtype_and_cast(tensor_a)
        tensor_b = self.check_dtype_and_cast(tensor_b)

        if tensor_a.track_gradient or tensor_b.track_gradient:
            track_gradient = True
        else:
            track_gradient = False

        super().__init__(track_gradient=track_gradient)

        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

        self.parents = [self.tensor_a, self.tensor_b]

    def get_value(self) -> np.ndarray:
        return self.tensor_a.numpy(), self.tensor_b.numpy()

    def _collapse_grad(
        self, t1: np.ndarray, t2: np.ndarray, to_collapse: np.ndarray
    ) -> np.ndarray:
        """Helps to collapse the gradient when dimensions do not match but
        forward operation broadcasted the input tensors.

        Parameters
        ----------
        t1 : numpy.ndarray
        t2 : numpy.ndarray
        to_collapse : numpy.ndarray
        """
        diff_dims = t1.ndim - t2.ndim
        for _ in range(diff_dims):
            to_collapse = to_collapse.sum(axis=0)

        if t2.size == 1:
            to_collapse = to_collapse.sum(keepdims=True)

        return to_collapse

    def _set_gradients(
        self, gradient_a: "Tensor", gradient_b: "Tensor"
    ) -> None:
        if self.tensor_a.track_gradient:

            if self.tensor_a.shape != gradient_a.shape:
                msg = (
                    f"Wrong gradient shape {gradient_a.shape} for a tensor of"
                    f" shape {self.tensor_a.shape}"
                )
                raise GradientShapeError(msg)

            if self.tensor_a.gradient is None:
                self.tensor_a.gradient = gradient_a
            else:
                self.tensor_a.gradient.value += gradient_a.value

        if self.tensor_b.track_gradient:

            if self.tensor_b.shape != gradient_a.shape:
                msg = (
                    f"Wrong gradient shape {gradient_b.shape} for a tensor of"
                    f" shape {self.tensor_b.shape}"
                )
                raise GradientShapeError(msg)

            if self.tensor_b.gradient is None:
                self.tensor_b.gradient = gradient_b
            else:
                self.tensor_b.gradient.value += gradient_b.value


class ReduceOp(UnaryOp):
    """Reduction operation."""

    def __init__(self, tensor: "Tensor"):
        super().__init__(tensor=tensor)


class OperationRunner:
    """Operation runner will take care of running an operation appropiately.

    In this context, "appropiately" means the following 2 steps:
        * The runner will compute the forward step of the operation contained
        in self.
        * The runner will update the parameter "track_gradient" of the output
        tensor and add a backward function if this parameter is True. The
        backward will correspond to the backward pass of the passed operation.

    The parameter "operation" must be a class, not an instance of a class.

    Parameters
    ----------
    opration : toydiff.Operation
        Operation to run.
    tensors : iterable of tensors
        Operands for the operation

    Example
    -------
    >>> import toydiff as tdf
    >>> tensor = tdf.Tensor([1, 2, 3, 4])
    >>> args, **kwargs = ...
    >>> out = tdf.OperationRunner(tdf.core.Add, tensor).run(*args, **kwargs)
    """

    __slots__ = ["operation"]

    def __init__(self, operation: Operation, *tensors):
        self.operation = operation(*tensors)

    def run(self, *args, **kwargs) -> "Tensor":
        operation = self.operation
        out = operation(*args, **kwargs)
        if operation.track_gradient:
            out.track_gradient = operation.track_gradient
            out.backward_fn = operation._backward_fn

        return out


# -----------------------------------------------------------------------------
# ----------------------------- BINARY OPERATIONS -----------------------------
# -----------------------------------------------------------------------------
class Add(BinaryOp):
    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        return Tensor(
            np.add(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        data_a, data_b = self.get_value()
        np_grad = gradient.numpy()

        grad_a = np_grad
        grad_b = np_grad

        if data_a.ndim > data_b.ndim:
            grad_b = self._collapse_grad(
                t1=data_a, t2=data_b, to_collapse=grad_b
            )

        elif data_b.ndim > data_a.ndim:
            grad_a = self._collapse_grad(
                t1=data_b, t2=data_a, to_collapse=grad_a
            )

        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Add(BinaryOp)"


def add(tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs) -> "Tensor":
    """Element-wise addition between 2 tensors.

    If any of the tensors involved in the addition has track_gradient=True, a
    backward function will be added to the output.

    Parameters
    ----------
    tensor_a : toydiff.Tensor
        Tensor to be added.
    tensor_b : toydiff.Tensor
        Tensor to be added.

    Returns
    -------
    out : toydiff.Tensor
        The sum of tensor_a and tensor_b, element-wise.
    """
    return OperationRunner(Add, tensor_a, tensor_b).run(*args, **kwargs)


def subtract(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Element-wise subtraction between 2 tensors.

    tensor_b will be subtracted from tensor_a.

    If any of the tensors involved in the subtraction has track_gradient=True,
    a backward function will be added to the output.

    Parameters
    ----------
    tensor_a : toydiff.Tensor
        Tensor to subtract from.
    tensor_b : toydiff.Tensor
        Subtracted tensor.

    Returns
    -------
    out : toydiff.Tensor
        The difference of tensor_a and tensor_b, element-wise.
    """
    return OperationRunner(Add, tensor_a, -tensor_b).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class MatrixMultiplication(BinaryOp):
    """Matrix multiplication operation class.

    It implements the forward and backward passes, but `toydiff.matmul`
    function should be used to compute the matrix product of two tensors, since
    it will take care of making the appropiate checks and set the gradients.
    """

    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        return Tensor(
            np.matmul(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        data_a, data_b = self.get_value()
        grad_np = gradient.numpy()
        gradient_a = Tensor(np.matmul(grad_np, data_b.T))
        gradient_b = Tensor(np.matmul(data_a.T, grad_np))
        self._set_gradients(gradient_a, gradient_b)

    def __repr__(self):
        return "MatrixMultiplication(BinaryOp)"


def matmul(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Matrix product of two tensors.

    Parameters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Return
    ------
    out : toydiff.Tensor
        Matrix product of the input tensors.
    """
    return OperationRunner(MatrixMultiplication, tensor_a, tensor_b).run(
        *args, **kwargs
    )


# -----------------------------------------------------------------------------
class FusedMultiplyAdd:
    def forward(self) -> "Tensor":
        pass

    def backward(self, gradient: "Tensor" = None) -> None:
        pass

    def __repr__(self) -> str:
        return "FusedMultiplyAdd(TernaryOp)"


def fma(
    tensor_a: "Tensor", tensor_b: "Tensor", tensor_c: "Tensor"
) -> "Tensor":
    """Fused matrix multiplication and addition operator.

    Currently, the operation is not performed by fusing the operations but by
    chaining them. Expect this to change in the future.
    """
    # TODO: https://github.com/nschloe/pyfma
    return matmul(tensor_a, tensor_b) + tensor_c


# -----------------------------------------------------------------------------
class Multiply(BinaryOp):
    def forward(self, *args, **kwargs):
        data_a, data_b = self.get_value()
        return Tensor(
            np.multiply(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        data_a, data_b = self.get_value()
        np_grad = gradient.numpy()

        grad_a = data_b * np_grad
        grad_b = data_a * np_grad

        if data_a.ndim > data_b.ndim:
            grad_b = self._collapse_grad(
                t1=data_a, t2=data_b, to_collapse=grad_b
            )

        elif data_b.ndim > data_a.ndim:
            grad_a = self._collapse_grad(
                t1=data_b, t2=data_a, to_collapse=grad_a
            )

        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Multiply(BinaryOp)"


def multiply(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Multiply two tensors element-wise.

    Paremeters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Returns
    -------
    toydiff.Tensor
    """
    return OperationRunner(Multiply, tensor_a, tensor_b).run(*args, **kwargs)


def divide(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Returns a true division of the inputs, element-wise.

    Operation performed is tensor_a / tensor_b. Output is achieved by combining
    multiply and power operations.

    Paremeters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Returns
    -------
    toydiff.Tensor
    """
    return OperationRunner(
        Multiply, tensor_a, power(tensor_b, Tensor(-1))
    ).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Power(BinaryOp):
    __slots__ = ["power"]
    def __init__(self, tensor_a: "Tensor", tensor_b: "Tensor"):
        super().__init__(tensor_a=tensor_a, tensor_b=tensor_b)
        self.power = None

    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        self.power = np.power(data_a, data_b, *args, **kwargs)
        return Tensor(
            self.power,
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        # data_a is base and data_b is exponent
        data_a, data_b = self.get_value()
        grad_np = gradient.numpy()

        grad_a = (data_b * np.power(data_a, data_b - 1)) * grad_np
        grad_b = (self.power * np.log(data_a)) * grad_np

        if data_a.ndim > data_b.ndim:
            grad_b = self._collapse_grad(
                t1=data_a, t2=data_b, to_collapse=grad_b
            )

        elif data_b.ndim > data_a.ndim:
            grad_a = self._collapse_grad(
                t1=data_b, t2=data_a, to_collapse=grad_a
            )

        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Power(BinaryOp)"


def power(tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs) -> "Tensor":
    """First tensor elements raised to powers from second tensor, element-wise.

    Parameters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Return
    ------
    out : toydiff.Tensor
        Power operation of the input tensors.
    """
    return OperationRunner(Power, tensor_a, tensor_b).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Maximum(BinaryOp):
    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        return Tensor(
            np.maximum(data_a, data_b, *args, **kwargs),
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        data_a, data_b = self.get_value()
        grad_np = gradient.numpy()
        base_grad = np.isclose(data_a, data_b) * 0.5

        grad_a = base_grad.copy()
        grad_a[data_a > data_b] = 1

        grad_b = base_grad
        grad_b[data_b > data_a] = 1

        if data_a.ndim > data_b.ndim:
            grad_a = grad_a * grad_np
            grad_b = self._collapse_grad(
                t1=data_a, t2=data_b, to_collapse=grad_b * grad_np
            )

        elif data_b.ndim > data_a.ndim:
            grad_b = grad_b * grad_np
            grad_a = self._collapse_grad(
                t1=data_b, t2=data_a, to_collapse=grad_a * grad_np
            )

        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Maximum(BinaryOp)"


def maximum(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Element-wise maximum of tensor elements.

    Compare two tensors and return a new tensor containing the element-wise
    maxima.

    Paremeters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Returns
    -------
    out : toydiff.Tensor
        The maximum of tensor_1 and tensor_b, element-wise.
    """
    return OperationRunner(Maximum, tensor_a, tensor_b).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Minimum(BinaryOp):
    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        return Tensor(
            np.minimum(data_a, data_b, *args, **kwargs),
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        data_a, data_b = self.get_value()
        grad_np = gradient.numpy()
        base_grad = np.isclose(data_a, data_b) * 0.5

        grad_a = base_grad.copy()
        grad_a[data_a < data_b] = 1

        grad_b = base_grad
        grad_b[data_b < data_a] = 1

        if data_a.ndim > data_b.ndim:
            grad_a = grad_a * grad_np
            grad_b = self._collapse_grad(
                t1=data_a, t2=data_b, to_collapse=grad_b * grad_np
            )

        elif data_b.ndim > data_a.ndim:
            grad_b = grad_b * grad_np
            grad_a = self._collapse_grad(
                t1=data_b, t2=data_a, to_collapse=grad_a * grad_np
            )

        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Minimum(BinaryOp)"


def minimum(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Element-wise minimum of tensor elements.

    Compare two tensors and return a new tensor containing the element-wise
    minima.

    Paremeters
    ----------
    tensor_a : toydiff.Tensor
    tensor_b : toydiff.Tensor

    Returns
    -------
    out : toydiff.Tensor
        The minimum of tensor_1 and tensor_b, element-wise.
    """
    return OperationRunner(Minimum, tensor_a, tensor_b).run(*args, **kwargs)


# -----------------------------------------------------------------------------
# ----------------------------- UNARY OPERATIONS ------------------------------
# -----------------------------------------------------------------------------
class Log(UnaryOp):
    def forward(self, *args, **kwargs):
        return Tensor(
            np.log(self.get_value(), *args, **kwargs),
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        self._set_gradients(Tensor(gradient.numpy() / self.get_value()))

    def __repr__(self) -> str:
        return "Log(UnaryOp)"


def log(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Element-wise natural logarithm.

    Parameters
    ----------
    tensor : toydiff.Tensor
    """
    return OperationRunner(Log, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Sigmoid(UnaryOp):
    __slots__ = ["sigmoid"]

    def __init__(self, tensor: "Tensor"):
        super().__init__(tensor)
        self.sigmoid = None  # avoid multiple computations

    def forward(self, *args, **kwargs):
        self.sigmoid = expit(self.get_value(), *args, **kwargs)
        return Tensor(
            self.sigmoid,
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        self._set_gradients(
            Tensor((self.sigmoid * (1 - self.sigmoid)) * gradient.numpy())
        )

    def __repr__(self) -> str:
        return "Sigmoid(UnaryOp)"


def sigmoid(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Computes the logistic sigmoid function of the elements of the input
    tensor.

    Paremters
    ---------
    tensor : toydiff.Tensor
        Tensor to apply the sigmoid to.

    Returns
    -------
    out : toydiff.Tensor
        Logistic sigmoid.
    """
    return OperationRunner(Sigmoid, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Negative(UnaryOp):
    def forward(self, *args, **kwargs):
        return Tensor(
            np.negative(self.get_value(), *args, **kwargs),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        self._set_gradients(
            Tensor(-np.ones_like(self.get_value()) * gradient.numpy())
        )

    def __repr__(self) -> str:
        return "Negative(UnaryOp)"


def negative(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Numerical negative, element-wise.

    Parameters
    ----------
    tensor : toydiff.Tensor
        Input tensor.

    Returns
    -------
    out : toydiff.Tensor
        Returned tensor.
    """
    return OperationRunner(Negative, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Sin(UnaryOp):
    def forward(self):
        return Tensor(
            np.sin(self.get_value()),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        self._set_gradients(
            Tensor(np.cos(self.get_value()) * gradient.numpy())
        )

    def __repr__(self) -> str:
        return "Sin(UnaryOp)"


def sin(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Sine element-wise.

    Parameters
    ----------
    tensor : toydiff.Tensor

    Return
    ------
    out : toydiff.Tensor
    """
    return OperationRunner(Sin, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Cos(UnaryOp):
    def forward(self):
        return Tensor(
            np.cos(self.get_value()),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        self._set_gradients(
            Tensor(-np.sin(self.get_value()) * gradient.numpy())
        )

    def __repr__(self) -> str:
        return "Cos(UnaryOp)"


def cos(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Cosine element-wise.

    Parameters
    ----------
    tensor : toydiff.Tensor

    Return
    ------
    out : toydiff.Tensor
    """
    return OperationRunner(Cos, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Tan(UnaryOp):
    def forward(self):
        return Tensor(
            np.tan(self.get_value()),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__()
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        self._set_gradients(
            Tensor(1 / (np.cos(self.get_value()) ** 2) * gradient.numpy())
        )

    def __repr__(self) -> str:
        return "Tan(UnaryOp)"


def tan(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Tangent function element-wise.

    Parameters
    ----------
    tensor : toydiff.Tensor

    Return
    ------
    out : toydiff.Tensor
    """
    return OperationRunner(Tan, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Reshape(UnaryOp):
    def forward(self, newshape, order="C"):
        return Tensor(
            np.reshape(self.get_value(), newshape=newshape, order=order),
            dtype=self.tensor.dtype,
            is_leaf=False,
            parents=self.parents,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        # original shape
        or_shape = self.tensor.numpy().shape
        self._set_gradients(
            Tensor(
                np.ones_like(
                    self.get_value()
                ) * gradient.numpy().reshape(or_shape)
            )
        )

    def __repr__(self) -> str:
        return "Reshape(UnaryOp)"


def reshape(
    tensor: "Tensor",
    newshape: Union[int, Tuple[int]],
    order: Literal["C", "F", "A"] = "C",
) -> "Tensor":
    """Gives a new shape to a Tensor without changing its data.

    Parameters
    ----------
    tensor : toydiff.Tensor
        Tensor to be reshaped.
    newshape : int or tuple or ints
        The new shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D tensor of that length. One shape
        dimension can be -1. In this case, the value is inferred from the
        length of the tensor and remaining dimensions.
    order : {"C", "F", "A"}, optional, default: "C"
        Read the elements of a using this index order, and place the elements
        into the reshaped tensor using this index order. 'C' means to read /
        write the elements using C-like index order, with the last axis index
        changing fastest, back to the first axis index changing slowest. 'F'
        means to read / write the elements using Fortran-like index order, with
        the first index changing fastest, and the last index changing slowest.

    Returns
    -------
    toydiff.Tensor
        This will be a new view object if possible; otherwise, it will be a
        copy. Note there is no guarantee of the memory layout (C- or Fortran-
        contiguous) of the returned tensor.
    """
    return OperationRunner(Reshape, tensor).run(newshape=newshape, order=order)


# -----------------------------------------------------------------------------
class Exponential(UnaryOp):
    def forward(self) -> "Tensor":
        return Tensor(
            np.exp(self.get_value()),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> "Tensor":
        self._set_gradients(Tensor(self.out.numpy() * gradient.numpy()))

    def __repr__(self):
        return "Exponential(UnaryOp)"


def exp(tensor: "Tensor") -> "Tensor":
    """Calculate the exponential of all elements in the input tensor."""
    return OperationRunner(Exponential, tensor).run()


# -----------------------------------------------------------------------------
class Transpose(UnaryOp):
    __slots__ = ["axes"]

    def __init__(self, tensor):
        super().__init__(tensor)
        self.axes = None

    def forward(self, axes=None):
        data = self.get_value()
        self.axes = axes
        return Tensor(
            np.transpose(data, axes=axes),
            dtype=self.tensor.dtype,
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        axes = self.axes
        data = self.get_value()

        if axes is None:
            grad_np = np.transpose(gradient.numpy())
        else:
            equivalence = dict(zip(axes, range(0, len(axes))))
            inverted_axes = [equivalence[ax] for ax in range(0, len(axes))]
            grad_np = np.transpose(gradient.numpy(), inverted_axes)

        self._set_gradients(Tensor(np.ones_like(data) * grad_np))

    def __repr__(self):
        return "Transpose(ReduceOp)"


def transpose(tensor: "Tensor", axes: tuple = None) -> "Tensor":
    """Returns a tensor with axes transposed.

    For a 1-D tensor, this returns an unchanged view of the original tensor, as
    a transposed vector is simply the same vector.
    """
    return OperationRunner(Transpose, tensor).run(axes=axes)


# -----------------------------------------------------------------------------
class Sign(UnaryOp):
    def forward(self):
        return Tensor(
            np.sign(self.get_value()),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__()
        )

    def backward(self, gradient: "Tensor" = None) -> None:
        self._set_gradients(
            Tensor(np.zeros_like(self.get_value()))
        )

    def __repr__(self):
        return "Sign(UnaryOp)"


def sign(tensor: "Tensor") -> "Tensor":
    """Element-wise sign function"""
    return OperationRunner(Sign, tensor).run()


# -----------------------------------------------------------------------------
class Absolute(UnaryOp):
    def forward(self):
        return Tensor(
            np.abs(self.get_value()),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__()
        )

    def backward(self, gradient: "Tensor" = None) -> None:
        self._set_gradients(gradient)

    def __repr__(self):
        return "Absolute(UnaryOp)"


def abs(tensor: "Tensor") -> "Tensor":
    """Element-wise absolute value function"""
    return OperationRunner(Absolute, tensor).run()


# -----------------------------------------------------------------------------
# ----------------------------- REDUCE OPERATIONS -----------------------------
# -----------------------------------------------------------------------------
class Max(ReduceOp):
    def forward(self, *args, **kwargs):
        return Tensor(
            np.max(self.get_value(), *args, **kwargs),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        # TODO: deal with axis != None
        data = self.get_value()
        indices = np.unravel_index(data.argmax(), data.shape)
        grad = np.zeros_like(data)
        grad[indices] = 1
        self._set_gradients(Tensor(grad * gradient.numpy()))

    def __repr__(self):
        return "Max(ReduceOp)"


def max(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Return the maximum along a given axis.

    Parameters
    ----------
    tensor : toydiff.Tensor
    """
    return OperationRunner(Max, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Min(ReduceOp):
    def forward(self, *args, **kwargs):
        return Tensor(
            np.min(self.get_value(), *args, **kwargs),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        # TODO: deal with axis != None
        data = self.get_value()
        indices = np.unravel_index(data.argmin(), data.shape)
        grad = np.zeros_like(data)
        grad[indices] = 1
        self._set_gradients(Tensor(grad * gradient.numpy()))

    def __repr__(self):
        return "Min(ReduceOp)"


def min(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Return the minimum along a given axis.

    Parameters
    ----------
    tensor : toydiff.Tensor
    """
    return OperationRunner(Min, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Sum(ReduceOp):
    def forward(self, *args, **kwargs):
        return Tensor(
            np.sum(self.get_value(), *args, **kwargs),
            parents=self.parents,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        self._set_gradients(
            Tensor(gradient.numpy() * np.ones_like(self.get_value()))
        )

    def __repr__(self):
        return "Sum(ReduceOp)"


def sum(tensor: "Tensor", *args, **kwargs) -> "Tensor":
    """Sum of tensor elements over a given axis.

    Parameters
    ----------
    tensor : toydiff.Tensor
        Elements to sum.

    Returns
    -------
    out : toydiff.Tensor
        Added elements.
    """
    return OperationRunner(Sum, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class Slice(ReduceOp):
    __slots__ = ["key"]

    def __init__(self, tensor):
        super().__init__(tensor)
        self.key = None

    def forward(self, key):
        self.key = key
        return Tensor(
            self.get_value().__getitem__(key),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        grad = np.zeros_like(self.tensor.numpy())
        grad.__setitem__(self.key, 1)
        self._set_gradients(Tensor(grad * gradient.numpy()))

    def __repr__(self):
        return "Slice(ReduceOp)"


# -----------------------------------------------------------------------------
class Mean(ReduceOp):
    __slots__ = ["axis"]
    def __init__(self, tensor: "Tensor"):
        super().__init__(tensor)
        self.axis = None

    def forward(self, axis=None, keepdims=False):
        self.axis = axis
        return Tensor(
            np.mean(self.get_value(), axis=axis, keepdims=keepdims),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )
    
    def backward(self, gradient: Optional["Tensor"] = None):
        data = self.get_value()
        shape = data.shape
        if self.axis is None:
            n = data.size
        else:
            n = shape[self.axis]

        # TODO: inefficient algorithm, find alternatives
        grad_np = gradient.numpy()
        ndims = data.ndim
        if ndims > 1 and self.out.ndim != 0:
            i = 1
            while i < (ndims - grad_np.ndim):
                grad_np = np.expand_dims(grad_np, i)
                i += 1

        self._set_gradients(Tensor(np.ones_like(data) * (grad_np / n)))

    def __repr__(self):
        return "Mean(ReduceOp)"


def mean(
    tensor: "Tensor", axis: Optional[int] = None, keepdims: bool = False
) -> "Tensor":
    """Compute the arithmetic mean along the specified axis.

    Parameters
    ----------
    tensor : toydiff.Tensor
    axis : int, optional, default: None
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
    keepdims : bool, optional, default: False
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array
    """
    return OperationRunner(Mean, tensor).run(axis=axis, keepdims=keepdims)


# -----------------------------------------------------------------------------
class StandardDeviation(ReduceOp):
    def forward(self, axis, ddof, keepdims):
        return Tensor(
            np.std(self.get_value(), axis=axis, keepdims=keepdims, ddof=ddof),
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )
    
    def backward(self, gradient: Optional["Tensor"] = None):
        raise NotImplementedError

    def __repr__(self):
        return "StandardDeviation(ReduceOp)"


def std(
    tensor: "Tensor",
    axis: Optional[int] = None,
    ddof: int = 0,
    keepdims: bool = False
):
    """
    return OperationRunner(StandardDeviation, tensor).run(
        axis=axis, keepdims=keepdims, ddof=ddof
    )
    """
    # TODO: much more faster to create a ReduceOp
    return power(power(tensor - tensor.mean(), 2).sum() / len(tensor), 0.5)


# -----------------------------------------------------------------------------
# ------------------------------ OTHER FUNCTIONS ------------------------------
# -----------------------------------------------------------------------------
def ones(
    shape: Tuple[int], dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.ones(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient
    )


def ones_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False,  **kwargs
) -> "Tensor":
    return Tensor(
        np.ones_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def zeros(
    shape: Tuple[int], dtype=np.int32, track_gradient=False,  **kwargs
) -> "Tensor":
    return Tensor(
        np.zeros(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient
    )


def zeros_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.zeros_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def empty(
    shape: Tuple[int], dtype=np.int32, track_gradient=False,  **kwargs
) -> "Tensor":
    return Tensor(
        np.empty(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def empty_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False,  **kwargs
) -> "Tensor":
    return Tensor(
        np.empty_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def rand(shape: Tuple[int], track_gradient: bool = False) -> "Tensor":
    """Random values in a given shape.

    Create a tensor of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).

    Parameters
    ----------
    shape : tuple of ints
        Shape of the generated tensor.
    track_gradient : bool, optional, default: False
        If True, the created tensor will be ready to track gradients.

    Returns
    -------
    toydiff.Tensor
        Generated tensor.
    """
    return Tensor(np.random.rand(*shape), track_gradient=track_gradient)


def randn(shape: Tuple[int], track_gradient: bool = False) -> "Tensor":
    """Return a sample (or samples) from the "standard normal" distribution.

    If positive int_like arguments are provided, randn generates a tensor of
    shape (d0, d1, ..., dn), filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1. A
    single float randomly sampled from the distribution is returned if no
    argument is provided.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the generated tensor.
    track_gradient : bool, optional, default: False
        If True, the created tensor will be ready to track gradients.

    Returns
    -------
    toydiff.Tensor
        Generated tensor.
    """
    return Tensor(np.random.randn(*shape), track_gradient=track_gradient)


# -----------------------------------------------------------------------------
# ------------------------------- Tensor Class --------------------------------
# -----------------------------------------------------------------------------
class Tensor:
    """A toydiff.Tensor is a multi-dimensional matrix containing elements of a
    single data type.

    Chaining tensors with arbitrary operations will generate a differentiable
    computational graph. Derivatives are computed using the chain rule. Only
    Tensors with `track_gradient=True` will receive a gradient Tensor once
    'backward' is called.

    Intermediate derivates are stored by default.

    Tensor creation
    ---------------
    You can create a tensor passing an array or an array-wrappable object:
    >>> import toydiff as tdf
    >>> import numpy as np
    >>> a = tdf.Tensor([1, 2, 3], track_gradient=True)
    >>> b = tdf.Tensor(np.random.rand(3, 3), track_gradient=True)

    ToyDiff also supports some functions to generate Tensors with ease:
    >>> tdf.rand((3,3), track_gradient=True)
    >>> tdf.zeros((3,3), track_gradient=True)
    >>> tdf.ones_like(a, track_gradient=True)

    Forward computation
    -------------------
    Simply operate like you would in NumPy
    >>> c = a + b

    `c` is now a non-leaf tensor whose parents are `a` and `b` and whose
    backward operation is the derivative of the addition function.

    We can add as many operations as we want:

    >>> d = tdf.log(c)
    >>> e = tdf.sum(d)

    Backward computation
    --------------------
    Calling backward on the last node of the graph will fill the gradients of
    all tensors in the graph that have `track_gradient=True`, storing them in
    an attribute called `gradient`:

    >>> e.backward()
    >>> print(a.gradient)

    Parameters
    ----------
    value : obj
        Object for the Tensor to wrap.
    is_leaf : bool, optional, default: True
        Whether this Tensor is a leaf node (no parents) or not.
    track_gradient : optional, default: False
        If True, gradients will be tracked for this tensor.
    parents : list of Tensor, optional, default: None
        Tensors that originated self. For example, the operation a + b = c will
        generate a Tensor `c` whose parents are `a` and `b`.
    op_name : str
        Name of the operation that generated self Tensor. Should be None for
        leaf tensors.

    Attributes
    ----------
    gradient : Tensor, default: None
        Gradient (of Tensor type) of self Tensor.
    backward_fn : callable, default: None
        Backward function of self Tensor if it has been created using an
        operation.
    """

    __slots__ = [
        "value",
        "dtype",
        "is_leaf",
        "track_gradient",
        "parents",
        "gradient",
        "backward_fn",
        "op_name",
        "__size",
        "__shape",
        "__ndim",
        "__backward_called",
    ]

    def __init__(
        self,
        value: object,
        dtype: Union[str, Type[np.dtype]] = np.float32,
        is_leaf: bool = True,
        track_gradient: bool = False,
        parents: List["Tensor"] = None,
        op_name: str = None,
    ):
        if op_name is not None and is_leaf:
            msg = f"An operation name '{op_name}' was given to a leaf tensor"
            raise ValueError(msg)

        self.value = np.array(value, dtype=dtype)
        self.dtype = dtype
        self.is_leaf = is_leaf
        self.track_gradient = track_gradient
        self.parents = parents
        self.op_name = op_name

        # attributes
        self.gradient: Tensor = None
        self.backward_fn: callable = None
        self.__size = self.value.size
        self.__shape = self.value.shape
        self.__ndim = self.value.ndim
        self.__backward_called = False

    def detach(self) -> "Tensor":
        """Detach tensor from the graph.

        Generates a new instance (leaf) Tensor with no gradients and no
        parents. The attribute `track_gradient` will be set to False.

        The internal numpy array will also be copied.

        Returns
        -------
        toydiff.Tensor
            Detached tensor.
        """
        return Tensor(self.value.copy(), dtype=self.dtype, is_leaf=True)

    def __iter__(self):
        return self.value.__iter__()  # should return tensors, not arrays

    def __setitem__(self, key, value):
        if self.__backward_called:
            msg = (
                "Cannot modify Tensor when backwad has been already called."
                " Use 'detach' method to generate a new instance with same"
                " properties but no gradient")
            raise InplaceModificationError(msg)

        # TODO: not sure if this is right
        self.value.__setitem__(key, value)

    def __getitem__(self, key):
        return OperationRunner(Slice, self).run(key)

    def __gt__(self, other):
        return Tensor(self.value > other.value)

    def __ge__(self, other):
        return Tensor(self.value >= other.value)

    def __lt__(self, other):
        return Tensor(self.value < other.value)

    def __le__(self, other):
        return Tensor(self.value <= other.value)

    def __neg__(self):
        return negative(self)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __rpow__(self, base):
        return power(base, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __rpow__(self, base):
        return power(base, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def matmul(self, other, *args, **kwargs) -> "Tensor":
        return matmul(self, other, *args, **kwargs)

    def max(self, *args, **kwargs) -> "Tensor":
        return max(self, *args, **kwargs)

    def maximum(self, other) -> "Tensor":
        return maximum(self, other)

    def min(self, *args, **kwargs) -> "Tensor":
        return min(self, *args, **kwargs)

    def minimum(self, other) -> "Tensor":
        return minimum(self, other)

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        return mean(self, axis=axis, keepdims=keepdims)

    def std(self, axis=None, ddof=0, keepdims=False) -> "Tensor":
        return std(self, axis=axis, ddof=ddof, keepdims=keepdims)

    def sum(self, *args, **kwargs) -> "Tensor":
        return sum(self, *args, **kwargs)

    def log(self, *args, **kwargs) -> "Tensor":
        """Calculate the natural log of all elements in self tensor."""
        return log(self, *args, **kwargs)

    def exp(self) -> "Tensor":
        """Calculate the exponential of all elements in self tensor."""
        return exp(self)

    def sigmoid(self, *args, **kwargs) -> "Tensor":
        return sigmoid(self, *args, **kwargs)

    def abs(self, *args, **kwargs) -> "Tensor":
        return abs(self, *args, **kwargs)

    def __repr__(self):
        rpr_ = self.value.__repr__().replace("array", "Tensor")[:-1]
        if self.op_name is not None:
            func_name = f"{self.op_name}.Backward"
            rpr_ = f"{rpr_}, backward_fn=<{func_name}>)"
        else:
            rpr_ = f"{rpr_}, track_gradient={self.track_gradient})"
        return rpr_

    def __len__(self):
        return len(self.value)

    @property
    def size(self):
        return self.__size

    @property
    def shape(self):
        return self.__shape

    @property
    def ndim(self):
        return self.__ndim

    @property
    def backward_called(self):
        """If True, the backward pass has already been called on this tensor"""
        return self.__backward_called

    @backward_called.setter
    def backward_called(self, val: bool):
        self.__backward_called = val

    @property
    def T(self):
        return transpose(self)

    def zero_grad(self) -> None:
        """Zeroes the gradient attribute of self tensor."""
        if self.track_gradient:
            self.gradient = Tensor(np.zeros_like(self.value, dtype=np.float32))
        else:
            msg = (
                "Trying to zero the gradient of a tensor with"
                " 'track_gradient=False'"
            )
            ZeroGradientError(msg)

    def numpy(self) -> np.ndarray:
        """Returns the internal numpy array."""
        return self.value

    def copy(self):  # also has backward
        return NotImplementedError

    def reshape(
        self,
        newshape: Union[int, Tuple[int]],
        order: Literal["C", "F", "A"] = "C"
    ) -> "Tensor":
        return reshape(self, newshape=newshape, order=order)

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Backward pass starting from self Tensor.

        The backward pass will sort topologically all tensors and fill their
        gradients in reverse order until reaching the leaf nodes.

        This function returns nothing; instead, the gradients of all tensors in
        the computational graph will be modified in-place.

        Parameters
        ----------
        gradient : toydiff.Tensor, optional, default: None
            Starting gradient. If None, a gradient Tensor of 1s and shape equal
            to self tensor shape will be passed.
        """
        if self.is_leaf:
            warn = (
                "Calling 'backward' on a leaf tensor will have no effect other"
                " than filling its gradient with ones"
            )
            warnings.warn(warn)

        if gradient is None:
            gradient = Tensor(np.ones_like(self.value))

        self.gradient = gradient
        sorted_tensors = topological_sort(self)
        for tensor in reversed(sorted_tensors):
            if tensor.is_leaf:
                continue

            if tensor.backward_fn is not None:
                tensor.backward_fn()
                tensor.backward_called = True
            else:
                msg = (
                    "Attempted to call 'backward' on a tensor with"
                    " 'backward_fn=None'"
                )
                raise NullBackwardFunctionError(msg)
