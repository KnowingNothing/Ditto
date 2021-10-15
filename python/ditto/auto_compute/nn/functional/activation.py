import tvm


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
    return tvm.te.compute(x.shape, _inner, name="ReLU")
