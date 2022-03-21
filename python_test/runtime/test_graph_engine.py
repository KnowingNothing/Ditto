import tvm
import pathlib
import json
import numpy as np
from tvm.target import Target
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto import runtime
from ditto.auto_compute.nn.model import resnet50, lenet5


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
    schedules = auto_schedule.retrieve_schedule_model([A], lenet5, schedule_option)
    built_mods = {}
    dev = tvm.cuda()
    for (key, (sch, args)) in schedules.items():
        mod = tvm.build(sch, args, target, target_host)
        built_mods[key] = mod

        arg_vs = []
        for arg in args:
            np_v = np.random.uniform(-1, 1, [int(x) for x in arg.shape]).astype(
                "float32"
            )
            tvm_v = tvm.nd.array(np_v, dev)
            arg_vs.append(tvm_v)
        mod(*arg_vs)

    model = lenet5()
    outputs = model(A)

    graph = ac.graph([A], outputs)
    ge = runtime.create_graph_engine(graph, built_mods, tvm.nd.device(target))
    fcompile = ge.get_function("compile")
    frun = ge.get_function("run")
    ftimeit = ge.get_function("timeit")
    fset_inputs = ge.get_function("set_inputs")
    fset_weight = ge.get_function("set_weight")
    fget_outputs = ge.get_function("get_outputs")
    fcompile()
    print("Compile done", flush=True)
    frun()
    print("Run done", flush=True)
    outputs = fget_outputs()
    print("Results:", flush=True)
    print(outputs[0].asnumpy())
    cost = ftimeit(10).value
    print("Time cost", cost, "ms")


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
