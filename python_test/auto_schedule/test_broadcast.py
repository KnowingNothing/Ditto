import tvm
from ditto import auto_compute as ac
from ditto import auto_schedule
from tvm import topi
import numpy as np
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
    def get_layer():
        A = tvm.te.placeholder([1, 32, 6, 6, 10])
        B = tvm.te.placeholder([1, 10, 10])
        C = tvm.te.compute(
            [1, 32, 6, 6, 10, 10],
            lambda n, c, h, w, p, q: A[n, c, h, w, p] * B[n, p, q],
        )
        D = tvm.te.placeholder([1, 32, 6, 6, 10, 10])
        r = tvm.te.reduce_axis([0, 10], "r")
        E = tvm.te.compute(
            [1, 32, 6, 6, 10],
            lambda n, c, h, w, p: tvm.te.sum(
                D[n, c, h, w, p, r] * B[n, p, r], axis=[r]
            ),
        )
        layer = ac.layer([C.op, E.op], inputs=[A, B, D])
        return [A, D, B, C, E]

    target = "cuda"
    target_host = "llvm"
    trials = 100
    task_name = "test"
    log_file = "tmp.log"
    builder = "local"
    runner = "local"

    schedule_option = auto_schedule.ScheduleOption(
        target,
        target_host=target_host,
        trials=trials,
        task_name=task_name,
        log_file=log_file,
        builder=builder,
        runner=runner,
    )

    sch, args = auto_schedule.auto_schedule(get_layer, schedule_option)


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
