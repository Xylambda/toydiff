from math import isclose
from toydiff.ops import *


# general variables
a = 2
b = -3


def test_add():
    op = Add()

    f_obt = op.forward(a, b)
    a_grad_obt, b_grad_obt = op.backward(incoming_grad=1)

    f_exp = -1
    a_grad_exp, b_grad_exp = 1, 1

    assert f_obt == f_exp, "Error in forward."
    assert a_grad_obt == a_grad_exp, "Error in a backward."
    assert b_grad_obt == b_grad_exp, "Error in b backward."


def test_mul():
    op = Multiply()

    f_obt = op.forward(a, b)
    a_grad_obt, b_grad_obt = op.backward(incoming_grad=1)

    f_exp = -6
    a_grad_exp, b_grad_exp = b, a

    assert f_obt == f_exp, "Error in forward."
    assert a_grad_obt == a_grad_exp, "Error in a backward."
    assert b_grad_obt == b_grad_exp, "Error in b backward."


def test_sub():
    op = Subtract()

    f_obt = op.forward(a, b)
    a_grad_obt, b_grad_obt = op.backward(incoming_grad=1)

    f_exp = 5
    a_grad_exp, b_grad_exp = 1, -1

    assert f_obt == f_exp, "Error in forward."
    assert a_grad_obt == a_grad_exp, "Error in a backward."
    assert b_grad_obt == b_grad_exp, "Error in b backward."


def test_pow():
    op = Pow()

    f_obt = op.forward(a, b)
    a_grad_obt, b_grad_obt = op.backward(incoming_grad=1)

    f_exp = 0.125
    a_grad_exp = -0.1875
    b_grad_exp = 0.08664339756999316

    assert f_obt == f_exp, "Error in forward."
    assert a_grad_obt == a_grad_exp, "Error in a backward."
    assert b_grad_obt == b_grad_exp, "Error in b backward."


def test_divide():
    op = Divide()

    f_obt = op.forward(a, b)
    a_grad_obt, b_grad_obt = op.backward(incoming_grad=1)

    f_exp = -0.6666666666666666
    a_grad_exp = -0.3333333333333333
    b_grad_exp = -0.2222222222222222

    assert isclose(f_obt, f_exp), "Error in forward."
    assert isclose(a_grad_obt, a_grad_exp), "Error in a backward."
    assert isclose(b_grad_obt, b_grad_exp), "Error in b backward."


def test_sin():
    op = Sin()

    f_obt = op.forward(a)
    a_grad_obt = op.backward(incoming_grad=1)

    f_exp = 0.9092974268256817
    a_grad_exp = -0.4161468365471424

    assert isclose(f_obt, f_exp), "Error in forward."
    assert isclose(a_grad_obt, a_grad_exp), "Error in a backward."


def test_cos():
    op = Cos()

    f_obt = op.forward(a)
    a_grad_obt = op.backward(incoming_grad=1)

    f_exp = -0.4161468365471424
    a_grad_exp = -0.9092974268256817

    assert isclose(f_obt, f_exp), "Error in forward."
    assert isclose(a_grad_obt, a_grad_exp), "Error in a backward."


def test_tan():
    op = Tan()

    f_obt = op.forward(a)
    a_grad_obt = op.backward(incoming_grad=1)

    f_exp = -2.185039863261519
    a_grad_exp = 5.774399204041917

    assert isclose(f_obt, f_exp), "Error in forward."
    assert isclose(a_grad_obt, a_grad_exp), "Error in a backward."
