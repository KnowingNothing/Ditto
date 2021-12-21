import pytest
import tvm
from ditto import utils


@pytest.mark.basic
def test_infer_range():
    a = tvm.tir.Var("a", "int32")
    b = tvm.tir.Var("b", "int32")
    c = tvm.tir.Var("c", "int32")
    d = tvm.tir.Var("d", "int32")
    
    expr1 = a + b
    expr2 = a * 2 + c
    expr3 = b - c
    expr4 = d + d * 3 - a
    
    range_map = {
        a: tvm.ir.Range(0, 32),
        b: tvm.ir.Range(0, 64),
        c: tvm.ir.Range(0, 3),
        d: tvm.ir.Range(0, 3)
    }
    
    ranges = utils.infer_range(
        [expr1, expr2, expr3, expr4],
        range_map
    )
    
    range1, range2, range3, range4 = ranges
    for r in ranges:
        print(r, r.extent.value + r.min.value)
    assert range1.extent == 95
    assert range2.extent == 31*2 + 2 + 1
    assert range3.extent == (63 - 0) - (0 - 2) + 1
    assert range4.extent == (2 * 4 - 0) - (0 - 31) + 1


if __name__ == "__main__":
    test_infer_range()