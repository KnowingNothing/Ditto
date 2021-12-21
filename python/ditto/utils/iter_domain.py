from . import _ffi_api
from typing import *
import tvm
from functools import reduce


def infer_range(expr_list: List[tvm.ir.PrimExpr],
                range_map: Dict[tvm.tir.Var, tvm.ir.Range]):
    var_expr_map = {}
    var_list = []
    for i, expr in enumerate(expr_list):
        var = tvm.tir.Var(f"v{i}", "int32")
        var_expr_map[var] = expr
        var_list.append(var)
    var_range_map = _ffi_api.InferRange(var_expr_map, range_map)

    ret = []
    for var in var_list:
        ret.append(var_range_map[var])
    return ret


def product(vlist, one=1):
    return list(reduce(lambda x, y: x * y, vlist, one))


def get_access_indices(op: tvm.te.tensor.Operation,
                       tensor: tvm.te.tensor.Tensor):
    ret = _ffi_api.GetAccessIndices(op, tensor.op)
    return [[xx for xx in x] for x in ret]
