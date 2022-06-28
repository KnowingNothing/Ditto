import tvm
from tvm.arith import Analyzer
from .pattern import PATTERN_SHUFFLE
from functools import reduce


def const_value(value, dtype):
    return tvm.tir.const(value, dtype)


def reshape(x, new_shape):
    """
    reshape
    ---
    args
    ---
    x: tvm.te.Tensor
    new_shape: List[int]

    Returns
    ---
    tvm.te.Tensor
    """
    shape = x.shape
    num_elems = reduce(lambda x, y: x * y, shape, 1)
    _num_elems = reduce(lambda x, y: x * y, new_shape, 1)
    ana = Analyzer()
    assert tvm.tir.analysis.expr_deep_equal(
        ana.simplify(num_elems), ana.simplify(_num_elems)
    )

    def _split(fused):
        indices = []
        strides = []
        strides.append(1)
        dim = len(shape)
        for i in range(1, dim):
            indices.append(fused // strides[-1] % shape[dim - i])
            strides.append(strides[-1] * shape[dim - i])
        indices.append(fused // strides[-1] % shape[0])
        indices = list(reversed(indices))
        return indices

    def _fuse(*idx):
        assert len(idx) == len(new_shape)
        dim = len(new_shape)
        index = idx[0]
        for i in range(1, dim):
            index = index * new_shape[i] + idx[i]
        return index

    return tvm.te.compute(new_shape, lambda *idx: x(*_split(_fuse(*idx))), name="reshape")


def transpose(x, new_indices_order):
    """
    transpose
    ---
    args
    ---
    x: tvm.te.Tensor
    new_indices_order: List[int]

    Returns
    ---
    tvm.te.Tensor
    """
    dim = len(x.shape)
    assert list(sorted(new_indices_order)) == list(range(dim)), f"{new_indices_order}"

    new_shape = []
    for v in new_indices_order:
        new_shape.append(x.shape[v])

    def _inner(*idx):
        indices = [0 for _ in idx]
        for i, v in enumerate(idx):
            indices[new_indices_order[i]] = v
        return indices

    return tvm.te.compute(
        new_shape, lambda *idx: x(*_inner(*idx)), name="transpose", tag=PATTERN_SHUFFLE
    )


def split(x, nparts, axis=-1):
    """
    split
    ---
    args
    ---
    x: tvm.te.Tensor
    nparts: int
    axis: int

    Returns
    ---
    tvm.te.Tensor
    """
    dim = len(x.shape)
    shape = [i for i in x.shape]
    while axis < 0:
        axis = dim + axis
    axis = axis % dim
    extent = shape[axis]//nparts
    shape[axis] = extent
    outputs = []
    assert nparts > 0
    
    def _inner(i, idx):
        ret = [j for j in idx]
        ret[axis] = ret[axis] + i * extent
        return ret
    
    for i in range(nparts):
        res = tvm.te.compute(
            shape, lambda *idx: x(*_inner(i, idx)), name="split", tag=PATTERN_SHUFFLE
        )
        outputs.append(res)
    return outputs