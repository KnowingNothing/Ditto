import tvm
from .pattern import PATTERN_SHUFFLE


def const_value(value, dtype):
    return tvm.tir.const(value, dtype)


def transpose(x, r, c):
    """
    ReLU
    ---
    args
    ---
    x: tvm.te.Tensor
    r: int
    c: int

    Returns
    ---
    tvm.te.Tensor
    """
    def _inner(*idx):
        indices = []
        for i, iv in enumerate(idx):
            if i == r:
                indices.append(idx[c])
            elif i == c:
                indices.append(idx[r])
            else:
                indices.append(iv)
        return x(*indices)
    return tvm.te.compute(x.shape, _inner, name="transpose", tag=PATTERN_SHUFFLE)

