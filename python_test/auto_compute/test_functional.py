import tvm
import pathlib
import json
import numpy as np
from tvm.target import Target
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto.auto_compute.nn.functional import transpose, reshape


from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[: len(prefix)] == prefix
    try:
        number = int(name[len(prefix) :])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)

        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix) :])


@register_test
def test1():
    A = tvm.te.placeholder([4, 5, 7, 8])
    B = transpose(A, [0, 2, 1, 3])
    print(B.op)


@register_test
def test2():
    A = tvm.te.placeholder([4, 5, 7, 8], dtype="float16")
    B = reshape(A, [20, 56])
    print(B.op)
    sch = tvm.te.create_schedule(B.op)
    func = tvm.build(sch, [A, B], "llvm")

    A_np = np.random.uniform(-10, 10, [4, 5, 7, 8]).astype("float16")
    B_np = np.random.uniform(-10, 10, [20, 56]).astype("float16")

    ctx = tvm.cpu(0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    func(A_tvm, B_tvm)

    from tvm import testing

    testing.assert_allclose(B_tvm.numpy(), np.reshape(A_np, [20, 56]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (str(args.case))
        case = TEST_CASES[args.case]
        case()
