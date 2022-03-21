import tvm
from ditto import auto_compute as ac
from ditto import autograd as ag
from ditto import neoflow as nf
from tvm import topi
import numpy as np
from ditto.auto_compute.nn.model import resnet50
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
    A = ac.layer_tensor([1, 3, 224, 224], dtype="float32", name="A")
    model = resnet50()
    outputs = model(A)

    graph = ac.graph([A], outputs)
    grad_graph = ag.grad_graph(graph, reserve_forward=True)
    print(grad_graph)
    fused_graph = nf.graph_fusion(graph)
    # print(fused_graph)
    grad_fused_graph = ag.grad_graph(fused_graph, reserve_forward=True)
    # print(grad_fused_graph)


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
