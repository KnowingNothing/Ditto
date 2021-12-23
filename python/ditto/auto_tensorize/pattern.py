"""Pattern of op"""
from .config import SUBSTANTIAL
from . import _ffi_api
from ditto import auto_compute as ac


def is_cubic(op, substantial=SUBSTANTIAL):
    """Judge if an op is cubic.
        Cubic op has both substantial outer-product
        and inner-product pattern.
        For example, a GEMM with large enough dimensions
        has both outer-product pattern and inner-product
        pattern.

    Args:
        op (tvm.tensor.Operation): the operation to judge
        substantial (int): the threshhold to judge if a dimension is large enough
    """
    return (_ffi_api.IsCubic(op, substantial) or ac.nn.pattern.PATTERN_CUBIC in op.tag)


def is_allred(op, substantial=SUBSTANTIAL):
    """Judge if an op is all reduce.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    return (_ffi_api.IsAllred(op, substantial) or ac.nn.pattern.PATTERN_ALLRED in op.tag)


def is_shuffle(op):
    """Judge if an op is shuffle.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    return (_ffi_api.IsShuffle(op) or ac.nn.pattern.PATTERN_SHUFFLE in op.tag)


def is_local(op, substantial=SUBSTANTIAL):
    """Judge if an op is local.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    return (_ffi_api.IsLocal(op, substantial) or ac.nn.pattern.PATTERN_LOCAL in op.tag)


def is_view(op):
    """Judge if an op is view.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    return (_ffi_api.IsView(op) or ac.nn.pattern.PATTERN_VIEW in op.tag)


def get_op_pattern(op):
    """Infer the pattern of one op.
        Cubic: substantial inner-product + outer-product

    Args:
        op (tvm.tensor.Operation): the operation to get pattern for

    Returns:
        pattern (str): PATTERN_XXX
    """
    if is_cubic(op):
        return ac.nn.pattern.PATTERN_CUBIC
    elif is_allred(op):
        return ac.nn.pattern.PATTERN_ALLRED
    elif is_shuffle(op):
        return ac.nn.pattern.PATTERN_SHUFFLE
    elif is_view(op):
        return ac.nn.pattern.PATTERN_VIEW
    elif is_local(op):
        return ac.nn.pattern.PATTERN_LOCAL
    else:
        raise ValueError(f"Can't judge the pattern of op:\n{op}.")
