import tvm
import pathlib
import json
import numpy as np
from tvm.target import Target
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto.auto_compute.nn.model import resnet50, lenet5, BERT


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
    A = ac.layer_tensor([1, 1, 32, 32], name="A", dtype="float32")
    target = "cuda"
    target_host = "llvm"
    trials = 100
    task_name = "test"
    log_file = "lenet_example_log.txt"
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
    schedules = auto_schedule.auto_schedule_model([A], lenet5, schedule_option)


@register_test
def test2():
    B = ac.layer_tensor([1, 1, 32, 32], name="B", dtype="float32")
    target = "cuda"
    target_host = "llvm"
    trials = 100
    task_name = "test"
    log_file = "lenet_example_log.txt"
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
    schedules = auto_schedule.retrieve_schedule_model([B], lenet5, schedule_option)


@register_test
def test3():
    X = ac.layer_tensor([16, 1024, 768], name="X", dtype="float32")
    model = BERT(dtype="float32")
    outputs = model(X)
    print(outputs)
    graph = ac.graph([X], [outputs])
    print(graph)


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
