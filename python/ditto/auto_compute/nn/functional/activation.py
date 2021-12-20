import tvm
from .pattern import PATTERN_LOCAL


def const_value(value, dtype):
    return tvm.tir.const(value, dtype)


def ReLU(x):
    """
    ReLU
    ---
    args
    ---
    x: tvm.te.Tensor

    Returns
    ---
    tvm.te.Tensor
    """
    zero = const_value(0, x.dtype)

    def _inner(*idx):
        return tvm.tir.if_then_else(x(*idx) > zero, x(*idx), zero)
    return tvm.te.compute(x.shape, _inner, name="ReLU", tag=PATTERN_LOCAL)


def GELU(x):
    def _c(i):
        return tvm.tir.const(i, x.dtype)

    def _gelu(*i):
        x_ = x(*i)
        y = x_ * _c(0.7978845608) * (_c(1.0) + _c(0.044715) * x_ * x_)
        y = _c(0.5) * x_ * (_c(1.0) + tvm.te.tanh(y))
        return y

    return tvm.te.compute(
        x.shape,
        _gelu,
        name="gelu",
        tag=PATTERN_LOCAL)
