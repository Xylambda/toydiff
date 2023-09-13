"""
Core of the library avagrad. It contains:
    1. A set of composable differentiable operations for avagrad.Tensor
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

    3. After each class, a function is created. Using this function will be
    enough to generate a computational graph from which to obtain the
    derivatives.

    4. The Tensor class is created using the above function and, if possible,
    dunder/magic methods are used to ensure a smooth use of the library.

"""
import warnings
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
from scipy.special import expit

from avagrad.exceptions import (
    GradientShapeError,
    InplaceModificationError,
    NullBackwardFunctionError,
    ZeroGradientError,
)
from avagrad.utils import gradient_collapse, topological_sort

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
    "cosh",
    "sinh",
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
    "bmm",
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
]

__TERNARY = ["fma"]

__all__ = (
    ["Tensor"]
    + __UNARY_OPS
    + __BINARY_OPS
    + __REDUCE_OPS
    + __OTHER
    + __TERNARY
)


class Operation(ABC):
    """Base class to create differentiable operations.

    To create an operation you must implement the forward and backward passes,
    but the forward computation must be calculated calling __call__ and not
    forward.

    Attributes
    ----------
    out : avagrad.Tensor
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
                msg = "Operations are supported only for avagrad.Tensor instances"
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
        avagrad.Tensor
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
        the backward method that is implemented for this operation using the
        aforementioned gradient.

        Parameters
        ----------
        gradient : avagrad.Tensor
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
        gradient : avagrad.Tensor, optional, default: None
            If None, a Tensor of 1's of the same shape as the output tensor of
            this operation will be used.
        """
        raise NotImplementedError("Subclasses must override this method")

    # TODO, DISC: nasty, nasty function ...
    def try_reshape(self, tensor, grad) -> None:
        """Nasty reshape function.

        After all gradient collapse, there can still be some cases where
        gradient tensor does not have the appropiate shape but its values are
        correct, e.g. (1,1) vs (1, ).

        This functions attemps a last reshape to match the associated tensor
        shape.

        Functions modifies gradient value in-place if shapes do not match and
        reshape can be done.
        """
        t_shape = tensor.shape
        grad_shape = grad.shape
        grad_val = grad.value

        if t_shape != grad_shape:
            try:
                grad.value = grad_val.reshape(t_shape)
            except:
                msg = (
                    f"Wrong gradient shape {grad_shape} for a tensor of shape"
                    f" {t_shape}"
                )
                raise GradientShapeError(msg)


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
            self.try_reshape(self.tensor, gradient)
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
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Attributes
    ----------
    parents : list of avagrad.Tensor
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

    def _set_gradients(
        self, gradient_a: "Tensor", gradient_b: "Tensor"
    ) -> None:
        if self.tensor_a.track_gradient:
            self.try_reshape(self.tensor_a, gradient_a)
            if self.tensor_a.gradient is None:
                self.tensor_a.gradient = gradient_a
            else:
                self.tensor_a.gradient.value += gradient_a.value

        if self.tensor_b.track_gradient:
            self.try_reshape(self.tensor_b, gradient_b)
            if self.tensor_b.gradient is None:
                self.tensor_b.gradient = gradient_b
            else:
                self.tensor_b.gradient.value += gradient_b.value


class ReduceOp(UnaryOp):
    """Reduction operation."""

    def __init__(self, tensor: "Tensor"):
        super().__init__(tensor=tensor)


class TernaryOp(Operation):  # where, fma
    """Base class to implement ternary operations.

    The method `get_value` will return the NumPy arrays of the `tensor_a`,
    `tensor_b` and `tensor_c` in the same order they were passed.

    Similarly, the method `_set_gradients` expects the first, second and thrid
    arguments to be the gradients for the first, second and third tensors
    passed in the constructors (respectively).

    Parameters
    ----------
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor
    tensor_c : avagrad.Tensor

    Attributes
    ----------
    parents : list of avagrad.Tensor
    """

    __slots__ = ["tensor_a", "tensor_b", "tensor_c", "parents"]

    def __init__(
        self, tensor_a: "Tensor", tensor_b: "Tensor", tensor_c: "Tensor"
    ):
        tensor_a = self.check_dtype_and_cast(tensor_a)
        tensor_b = self.check_dtype_and_cast(tensor_b)
        tensor_c = self.check_dtype_and_cast(tensor_c)

        if (
            tensor_a.track_gradient
            or tensor_b.track_gradient
            or tensor_c.track_gradient
        ):
            track_gradient = True
        else:
            track_gradient = False

        super().__init__(track_gradient=track_gradient)

        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.tensor_c = tensor_c

        self.parents = [self.tensor_a, self.tensor_b, self.tensor_c]

    def get_value(self) -> np.ndarray:
        return (
            self.tensor_a.numpy(),
            self.tensor_b.numpy(),
            self.tensor_c.numpy(),
        )

    def _set_gradients(
        self,
        gradient_a: "Tensor",
        gradient_b: "Tensor",
        gradient_c: "Tensor",
    ) -> None:
        if self.tensor_a.track_gradient:
            self.try_reshape(self.tensor_a, gradient_a)
            if self.tensor_a.gradient is None:
                self.tensor_a.gradient = gradient_a
            else:
                self.tensor_a.gradient.value += gradient_a.value

        if self.tensor_b.track_gradient:
            self.try_reshape(self.tensor_b, gradient_b)
            if self.tensor_b.gradient is None:
                self.tensor_b.gradient = gradient_b
            else:
                self.tensor_b.gradient.value += gradient_b.value

        if self.tensor_c.track_gradient:
            self.try_reshape(self.tensor_c, gradient_c)
            if self.tensor_c.gradient is None:
                self.tensor_c.gradient = gradient_c
            else:
                self.tensor_c.gradient.value += gradient_c.value


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
    opration : avagrad.Operation
        Operation to run.
    tensors : iterable of tensors
        Operands for the operation

    Example
    -------
    >>> from avagrad.core import Sum, OperationRunner, Tensor
    >>> tensor = ag.Tensor([1, 2, 3, 4])
    >>> args, kwargs = ...
    >>> out = OperationRunner(Sum, tensor).run(*args, **kwargs)
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
# ----------------------------- TERNARY OPERATIONS ----------------------------
# -----------------------------------------------------------------------------
class Where(TernaryOp):
    def forward(self, *args, **kwargs) -> "Tensor":
        return super().forward(*args, **kwargs)

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        pass


# -----------------------------------------------------------------------------
class FusedMatMulAdd(TernaryOp):
    __slots__ = ["mm"]

    def __init__(
        self, tensor_a: "Tensor", tensor_b: "Tensor", tensor_c: "Tensor"
    ):
        super().__init__(tensor_a, tensor_b, tensor_c)
        self.mm = None

    def forward(self) -> "Tensor":
        data_a, data_b, data_c = self.get_value()
        self.mm = np.matmul(data_a, data_b)
        return Tensor(
            self.mm + data_c,
            is_leaf=False,
            track_gradient=self.track_gradient,
            parents=self.parents,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        data_a, data_b, data_c = self.get_value()
        grad_np = gradient.numpy()

        grad_a = Tensor(np.matmul(grad_np, data_b.T))
        grad_b = Tensor(np.matmul(data_a.T, grad_np))

        # consider a the matmul and b the tensor c
        _, grad_c = gradient_collapse(self.mm, data_c, self.mm, grad_np)
        self._set_gradients(grad_a, grad_b, Tensor(grad_c))

    def __repr__(self):
        return "FusedMatMulAdd(TernaryOp)"


def fma(
    tensor_a: "Tensor", tensor_b: "Tensor", tensor_c: "Tensor"
) -> "Tensor":
    """Fused matrix multiplication and addition operator.

    Performs a matrix multiplication of `tensor_a` and `tensor_b` and adds the
    result to `tensor_c`.

    Parameters
    ----------
    tensor_a : avagrad.Tensor
        Tensor A of the matrix multiplication A x B.
    tensor_b : avagrad.Tensor
        Tensor B of the matrix multiplication A x B.
    tensor_c : avagrad.Tensor
        Tensor C of the operation (A x B) + C

    Returns
    -------
    avagrad.Tensor
        Output tensor.

    Warning
    -------
    Currently, this operation is not performed by fusing the operations but by
    chaining them in NumPy: np.matmul(a, b) + c
    """
    return OperationRunner(FusedMatMulAdd, tensor_a, tensor_b, tensor_c).run()


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
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        data_a, data_b = self.get_value()
        np_grad = gradient.numpy()

        grad_a, grad_b = gradient_collapse(data_a, data_b, np_grad, np_grad)
        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Add(BinaryOp)"


def add(tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs) -> "Tensor":
    """Element-wise addition between 2 tensors.

    If any of the tensors involved in the addition has track_gradient=True, a
    backward function will be added to the output.

    Parameters
    ----------
    tensor_a : avagrad.Tensor
        Tensor to be added.
    tensor_b : avagrad.Tensor
        Tensor to be added.

    Returns
    -------
    out : avagrad.Tensor
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
    tensor_a : avagrad.Tensor
        Tensor to subtract from.
    tensor_b : avagrad.Tensor
        Subtracted tensor.

    Returns
    -------
    out : avagrad.Tensor
        The difference of tensor_a and tensor_b, element-wise.
    """
    return OperationRunner(Add, tensor_a, -tensor_b).run(*args, **kwargs)


# -----------------------------------------------------------------------------
class MatrixMultiplication(BinaryOp):
    """Matrix multiplication operation class.

    It implements the forward and backward passes, but `avagrad.matmul`
    function should be used to compute the matrix product of two tensors, since
    it will take care of making the appropiate checks and set the gradients.
    """

    def forward(self, *args, **kwargs) -> "Tensor":
        data_a, data_b = self.get_value()
        return Tensor(
            np.matmul(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
            track_gradient=self.track_gradient,
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
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
        Matrix product of the input tensors.
    """
    return OperationRunner(MatrixMultiplication, tensor_a, tensor_b).run(
        *args, **kwargs
    )


# -----------------------------------------------------------------------------
class BatchMatrixMultiplication(BinaryOp):
    def __init__(
        self, tensor_a: "Tensor", tensor_b: "Tensor", tensor_c: "Tensor"
    ):
        if tensor_a.ndim != 3:
            raise Exception("'tensor_a' must be a 3D tensor")

        if tensor_b.ndim != 3:
            raise Exception("'tensor_b' must be a 3D tensor")

        super().__init__(tensor_a, tensor_b, tensor_c)

    def forward(self):
        data_a, data_b = self.get_value()
        # np.stack([a[i] @ b[i] for i in range(a.shape[0])])
        return Tensor(
            np.eisum("ijk, ikz -> ijz", data_a, data_b),
            parents=self.parents,
            is_leaf=False,
            track_gradient=self.track_gradient,
            is_leaf=False,
            op_name=self.__repr__(),
        )

    def backward(self, gradient=None):
        data_a, data_b = self.get_value()
        grad_np = gradient.numpy()
        grad_a = np.einsum(
            "ijk, ikz -> ijz",
            grad_np,
            np.transpose(data_b.numpy(), (0, 2, 1)),
        )
        grad_b = np.einsum(
            "ijk, ikz -> ijz", np.transpose(data_a, (0, 2, 1)), grad_np
        )
        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "BatchMatrixMultiplication(BinaryOp)"


def bmm(tensor_a: "Tensor", tensor_b: "Tensor") -> "Tensor":
    """Batch matrix-matrix product of 2 tensors.

    Both `tensor_a` and `tensor_b` must be 3D tensors.

    Paremeters
    ----------
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Returns
    -------
    avagrad.Tensor
    """
    return Operation(BatchMatrixMultiplication, tensor_a, tensor_b).run()


# -----------------------------------------------------------------------------
class Multiply(BinaryOp):
    def forward(self, *args, **kwargs):
        data_a, data_b = self.get_value()
        return Tensor(
            np.multiply(data_a, data_b, *args, **kwargs),
            parents=self.parents,
            is_leaf=False,
            track_gradient=self.track_gradient,
            op_name=self.__repr__(),
        )

    def backward(self, gradient: Optional["Tensor"] = None):
        data_a, data_b = self.get_value()
        np_grad = gradient.numpy()

        grad_a = data_b * np_grad
        grad_b = data_a * np_grad

        grad_a, grad_b = gradient_collapse(data_a, data_b, grad_a, grad_b)
        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Multiply(BinaryOp)"


def multiply(
    tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs
) -> "Tensor":
    """Multiply two tensors element-wise.

    Paremeters
    ----------
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Returns
    -------
    avagrad.Tensor
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
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Returns
    -------
    avagrad.Tensor
    """
    return OperationRunner(Multiply, tensor_a, power(tensor_b, -1)).run(
        *args, **kwargs
    )


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

        grad_a, grad_b = gradient_collapse(data_a, data_b, grad_a, grad_b)
        self._set_gradients(Tensor(grad_a), Tensor(grad_b))

    def __repr__(self):
        return "Power(BinaryOp)"


def power(tensor_a: "Tensor", tensor_b: "Tensor", *args, **kwargs) -> "Tensor":
    """First tensor elements raised to powers from second tensor, element-wise.

    Parameters
    ----------
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
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
        grad_a = grad_a * grad_np

        grad_b = base_grad
        grad_b[data_b > data_a] = 1
        grad_b = grad_b * grad_np

        grad_a, grad_b = gradient_collapse(data_a, data_b, grad_a, grad_b)
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
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Returns
    -------
    out : avagrad.Tensor
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
        grad_a = grad_a * grad_np

        grad_b = base_grad
        grad_b[data_b < data_a] = 1
        grad_b = grad_b * grad_np

        grad_a, grad_b = gradient_collapse(data_a, data_b, grad_a, grad_b)
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
    tensor_a : avagrad.Tensor
    tensor_b : avagrad.Tensor

    Returns
    -------
    out : avagrad.Tensor
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
    tensor : avagrad.Tensor
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
    tensor : avagrad.Tensor
        Tensor to apply the sigmoid to.

    Returns
    -------
    out : avagrad.Tensor
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
    tensor : avagrad.Tensor
        Input tensor.

    Returns
    -------
    out : avagrad.Tensor
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
    tensor : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
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
    tensor : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
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
            op_name=self.__repr__(),
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
    tensor : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
    """
    return OperationRunner(Tan, tensor).run(*args, **kwargs)


# -----------------------------------------------------------------------------
def cosh(tensor: "Tensor") -> "Tensor":
    """Element-wise hyperbolic cosine function.

    Parameters
    ----------
    tensor : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
    """
    return (tensor.exp() + (-tensor).exp()) / 2


def sinh(tensor: "Tensor") -> "Tensor":
    """Element-wise hyperbolic sine function.

    Parameters
    ----------
    tensor : avagrad.Tensor

    Return
    ------
    out : avagrad.Tensor
    """
    return (tensor.exp() - (-tensor).exp()) / 2


# -----------------------------------------------------------------------------
class Reshape(UnaryOp):
    def forward(self, newshape, order="C"):
        return Tensor(
            np.reshape(self.get_value(), newshape=newshape, order=order),
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
                np.ones_like(self.get_value())
                * gradient.numpy().reshape(or_shape)
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
    tensor : avagrad.Tensor
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
    avagrad.Tensor
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
            op_name=self.__repr__(),
        )

    def backward(self, gradient: "Tensor" = None) -> None:
        self._set_gradients(Tensor(np.zeros_like(self.get_value())))

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
            op_name=self.__repr__(),
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
    tensor : avagrad.Tensor
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
    tensor : avagrad.Tensor
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
    tensor : avagrad.Tensor
        Elements to sum.

    Returns
    -------
    out : avagrad.Tensor
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
    tensor : avagrad.Tensor
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
    keepdims: bool = False,
):
    """
    return OperationRunner(StandardDeviation, tensor).run(
        axis=axis, keepdims=keepdims, ddof=ddof
    )
    """
    # TODO: it will probably be much faster to create a ReduceOp
    return power(
        power(tensor - tensor.mean(axis=axis, keepdims=keepdims), 2).sum()
        / (len(tensor) - ddof),
        0.5,
    )


# -----------------------------------------------------------------------------
# ------------------------------ OTHER FUNCTIONS ------------------------------
# -----------------------------------------------------------------------------
def ones(
    shape: Tuple[int], dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.ones(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def ones_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.ones_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def zeros(
    shape: Tuple[int], dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.zeros(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def zeros_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.zeros_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def empty(
    shape: Tuple[int], dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.empty(shape=shape, dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


def empty_like(
    tensor: "Tensor", dtype=np.int32, track_gradient=False, **kwargs
) -> "Tensor":
    return Tensor(
        np.empty_like(tensor.numpy(), dtype=dtype, **kwargs),
        track_gradient=track_gradient,
    )


# -----------------------------------------------------------------------------
# ------------------------------- Tensor Class --------------------------------
# -----------------------------------------------------------------------------
class Tensor:
    """A avagrad.Tensor is a multi-dimensional matrix containing elements of a
    single data type.

    Chaining tensors with arbitrary operations will generate a differentiable
    computational graph. Derivatives are computed using the chain rule. Only
    Tensors with `track_gradient=True` will receive a gradient Tensor once
    'backward' is called.

    Intermediate derivates are stored by default.

    Tensor creation
    ---------------
    You can create a tensor passing an array or an array-wrappable object:
    >>> import avagrad as ag
    >>> import numpy as np
    >>> a = ag.Tensor([1, 2, 3], track_gradient=True)
    >>> b = ag.Tensor(np.random.rand(3, 3), track_gradient=True)

    avagrad also supports some functions to generate Tensors with ease:
    >>> ag.rand((3,3), track_gradient=True)
    >>> ag.zeros((3,3), track_gradient=True)
    >>> ag.ones_like(a, track_gradient=True)

    Forward computation
    -------------------
    Simply operate like you would in NumPy
    >>> c = a + b

    `c` is now a non-leaf tensor whose parents are `a` and `b` and whose
    backward operation is the derivative of the addition function.

    We can add as many operations as we want:

    >>> d = ag.log(c)
    >>> e = ag.sum(d)

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
        avagrad.Tensor
            Detached tensor.
        """
        return Tensor(self.value.copy(), dtype=self.dtype, is_leaf=True)

    def __iter__(self):
        return self.value.__iter__()  # should return tensors, not arrays

    def __setitem__(self, key, value):
        if self.__backward_called:
            msg = (
                "Can't modify Tensor when backwad has been already called. Use"
                " 'detach' method to generate a new instance with same"
                " properties but no gradient"
            )
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

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

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
        """Matrix multiplication between self and passed tensor"""
        return matmul(self, other, *args, **kwargs)

    def bmm(self, other) -> "Tensor":
        """Batch matrix multiplication between self and passed tensor"""
        return bmm(self, other)

    def max(self, *args, **kwargs) -> "Tensor":
        """Maximum of self tensor along given axis."""
        return max(self, *args, **kwargs)

    def maximum(self, other) -> "Tensor":
        """Element-wise maximum operation between self and passed tensor."""
        return maximum(self, other)

    def min(self, *args, **kwargs) -> "Tensor":
        """Minimum of self tensor along given axis."""
        return min(self, *args, **kwargs)

    def minimum(self, other) -> "Tensor":
        """Element-wise minimum operation between self and passed tensor."""
        return minimum(self, other)

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        """Compute the arithmetic mean along the specified axis."""
        return mean(self, axis=axis, keepdims=keepdims)

    def std(self, axis=None, ddof=0, keepdims=False) -> "Tensor":
        """Compute the standard deviation along the specified axis."""
        return std(self, axis=axis, ddof=ddof, keepdims=keepdims)

    def sum(self, *args, **kwargs) -> "Tensor":
        """Compute sum along the specified axis."""
        return sum(self, *args, **kwargs)

    def log(self, *args, **kwargs) -> "Tensor":
        """Compute the natural log of all elements in self tensor."""
        return log(self, *args, **kwargs)

    def exp(self) -> "Tensor":
        """Compute the exponential of all elements in self tensor."""
        return exp(self)

    def sigmoid(self, *args, **kwargs) -> "Tensor":
        """Compute sigmoid for all elements in self tensor."""
        return sigmoid(self, *args, **kwargs)

    def abs(self, *args, **kwargs) -> "Tensor":
        """Compute absolute value for all elements in self tensor."""
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

    def zero_grad(self, criterion="none") -> None:
        """Zeroes the gradient attribute of self tensor.

        By default val="none" because assigning variables is more efficient
        than adding values (this will happen when calling backward, since the
        library accumulates gradients by default).

        Parameters
        ----------
        criterion : str, optional, default: "none"
            If "none", gradient tensor will be set to None. If "zero", gradient
            tensor will be a Tensor of zeros.
        """
        if self.track_gradient:
            if criterion == "zero":
                self.gradient = Tensor(
                    np.zeros_like(self.value, dtype=self.dtype)
                )
            elif criterion == "none":
                self.gradient = None
            else:
                ValueError(f"Unsupported criterion parameter: '{criterion}'")
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
        order: Literal["C", "F", "A"] = "C",
    ) -> "Tensor":
        return reshape(self, newshape=newshape, order=order)

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Backward pass starting from self Tensor.

        The backward pass will sort topologically all tensors and fill their
        gradients in reverse order until reaching the leaf nodes.

        This function returns nothing; instead, the gradients of all tensors in
        the computational graph will be modified in-place (for tensors with
        `track_gradient=True`).

        Parameters
        ----------
        gradient : avagrad.Tensor, optional, default: None
            Starting gradient. If None, a gradient Tensor of 1s and shape equal
            to self tensor shape will be passed.
        """
        if self.is_leaf:
            warn = (
                "Calling 'backward' on a leaf tensor will have no effect other"
                " than filling its gradient with ones or passed gradient"
            )
            warnings.warn(warn)

        if gradient is None:
            gradient = ones_like(self)

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
