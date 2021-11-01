import tvm
from ditto import auto_compute as ac
from tvm import topi
import numpy as np
from ditto.auto_compute.nn.model import resnet50
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


def _2mm(M, L, K, N):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, L], name="B", dtype="float32")
    C = tvm.te.placeholder([L, N], name="C", dtype="float32")
    D = tvm.te.placeholder([M, N], name="D", dtype="float32")

    rk = tvm.te.reduce_axis([0, K], name="rk")
    tmp = tvm.te.compute(
        [M, L],
        lambda m, l:
            tvm.te.sum(A[m, rk] * B[rk, l], axis=rk),
        name="tmp"
    )

    rl = tvm.te.reduce_axis([0, L], name="rl")
    E = tvm.te.compute(
        [M, N],
        lambda m, n:
            tvm.te.sum(tmp[m, rl] * C[rl, n], axis=rl),
        name="E"
    )
    F = tvm.te.compute(
        [M, N],
        lambda m, n:
            0.5 * E[m, n] + 0.5 * D[m, n],
        name="F"
    )
    return [F], [A, B, C, D]


@register_test
def test1():
    outs, ins = _2mm(512, 512, 512, 512)
    F = outs[0]
    A, B, C, D = ins
    layer = ac.layer(F.op, inputs=[A, B], weights=[C, D], name="2mm")

    lA = ac.layer_tensor([512, 512], name="lA")
    lB = ac.layer_tensor([512, 512], name="lB")
    lF = layer(lA, lB)
    graph = ac.graph([lA, lB], [lF])
    print(graph)

    state = ac.create_graph_state(graph)
    layers = state.normalize_partition_layer(layer)
    for l in layers:
        print(l)
    graph = state.make_compute([lA, lB])
    print(graph)


@register_test
def test2():
    A = ac.layer_tensor([1, 1, 32, 32], dtype="float32", name="A")
    model = resnet50()
    outputs = model(A)

    graph = ac.graph([A], outputs)
    print(graph)
    state = ac.create_graph_state(graph)
    for layer in graph.all_layers:
        layers = state.normalize_partition_layer(layer)
    graph = state.make_compute([A])
    print(graph)


@register_test
def test3():
    outs, ins = _2mm(512, 512, 512, 512)
    F = outs[0]
    A, B, C, D = ins
    layer = ac.layer(F.op, inputs=[A, B], weights=[C, D], name="2mm")

    lA = ac.layer_tensor([512, 512], name="lA")
    lB = ac.layer_tensor([512, 512], name="lB")
    lF = layer(lA, lB)
    graph = ac.graph([lA, lB], [lF])
    print(graph)

    state = ac.create_graph_state(graph)
    l1, l2, l3 = state.normalize_partition_layer(layer)
    l = state.fuse_layer(l1, l3)
    graph = state.make_compute([lA, lB])
    print(graph)


@register_test
def test4():
    A = ac.layer_tensor([1, 1, 32, 32], dtype="float32", name="A")
    model = resnet50()
    outputs = model(A)

    graph = ac.graph([A], outputs)
    print(graph)
    state = ac.create_graph_state(graph)
    for i, layer in enumerate(graph.all_layers):
        # if i == 1:
        layers = state.normalize_partition_layer(layer)
        print()
        print("################################3", flush=True)
        for l in layers:
            print(l, flush=True)
        print("################################3", flush=True)
        l = state.fuse_layer(layers[0], layers[-1])
        print(l)
        print("################################3", flush=True)
    graph = state.make_compute([A])
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
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
