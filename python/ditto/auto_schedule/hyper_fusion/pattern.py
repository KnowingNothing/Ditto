"""Pattern of op"""


# the patterns of loop nests
PATTERN_CUBIC = "PATTERN_CUBIC"
PATTERN_ALLRED = "PATTERN_ALLRED"
PATTERN_SHUFFLE = "PATTERN_SHUFFLE"
PATTERN_LOCAL = "PATTERN_LOCAL"
PATTERN_VIEW = "PATTERN_VIEW"


def is_cubic(op, substantial=16):
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
    raise NotImplementedError()


def is_allred(op):
    """Judge if an op is all reduce.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    raise NotImplementedError()


def is_shuffle(op):
    """Judge if an op is shuffle.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    raise NotImplementedError()


def is_local(op):
    """Judge if an op is local.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    raise NotImplementedError()


def is_view(op):
    """Judge if an op is view.

    Args:
        op (tvm.tensor.Operation): the operation to judge
    """
    raise NotImplementedError()


def get_op_pattern(op):
    """Infer the pattern of one op.
        Cubic: substantial inner-product + outer-product

    Args:
        op (tvm.tensor.Operation): the operation to get pattern for

    Returns:
        pattern (str): PATTERN_XXX
    """
    if is_cubic(op):
        return PATTERN_CUBIC
    elif is_allred(op):
        return PATTERN_ALLRED
    elif is_shuffle(op):
        return PATTERN_SHUFFLE
    elif is_local(op):
        return PATTERN_LOCAL
    elif is_view(op):
        return PATTERN_VIEW
    else:
        raise ValueError(f"Can't judge the pattern of op:\n{op}.")