import tvm
from ditto import auto_compute as ac
from tvm import topi
import numpy as np
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


@register_test
def test1():
    x = tvm.te.placeholder((32, 3, 28, 28), name='xx')
    w1 = tvm.te.placeholder((10, 3, 3, 3), name='w1')
    w2 = tvm.te.placeholder((10, 10, 3, 3), name='w2')
    z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
    z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
    y = topi.sum(z2)

    # make a layer
    layer = ac.layer(y.op, inputs=[x], weights=[w1, w2])
    new_x = ac.layer_tensor((32, 3, 28, 28), name='new_x')
    new_y = layer(new_x)
    block = ac.block(new_y)

def _2mm(M, L, K, N):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, L], name="B", dtype="float32")
    C = tvm.te.placeholder([L, N], name="C", dtype="float32")
    D = tvm.te.placeholder([M, N], name="D", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")

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
            alpha * E[m, n] + beta * D[m, n],
        name="F"
    )
    return [F], [A, B, C, D, alpha, beta]


@register_test
def test2():
    M, L, K, N = 100, 40, 50, 60
    outs, ins = _2mm(M, L, K, N)
    F, = outs
    A, B, C, D, alpha, beta = ins
    layer = ac.layer(F.op, inputs=[A], weights=[
                        B, C, D], const_scalars=[alpha, beta])
    layer_state = ac.create_layer_state(layer)

    out = layer.ops[0]
    sch = tvm.te.create_schedule(out)
    all_tensors = [*layer.inputs,
                   *layer.weights,
                   *layer.const_scalars,
                   *layer.const_tensors,
                   out.output(0)]
    print("Original IR")
    print(tvm.lower(sch, all_tensors, simple_mode=True))
    old_func = tvm.build(sch, all_tensors, "llvm")
    A_np = np.random.uniform(-10, 10, [M, K]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, L]).astype("float32")
    C_np = np.random.uniform(-10, 10, [L, N]).astype("float32")
    D_np = np.random.uniform(-10, 10, [M, N]).astype("float32")
    F_np = np.random.uniform(-10, 10, [M, N]).astype("float32")
    ctx = tvm.cpu(0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)
    D_tvm = tvm.nd.array(D_np, ctx)
    F_tvm = tvm.nd.array(F_np, ctx)
    old_func(A_tvm, B_tvm, C_tvm, D_tvm, 1.0, 1.0, F_tvm)
    
    # transform the compute
    E = F.op.input_tensors[0]
    tmp = E.op.input_tensors[0]

    m, k = layer_state[A].axis()
    trans_A = layer_state.transform(
        A,
        [m.var//16, k.var//16, m.var%16, k.var%16],
        lambda m1, k1, m2, k2: [m1*16 + m2, k1*16 + k2],
        [],
        lambda : None)
    
    m, k = layer_state[trans_A].axis()
    trans_trans_A = layer_state.implicit_fold(trans_A, m, factor=7)
    m1, m2, k = layer_state[trans_trans_A].axis()
    trans_trans_trans_A = layer_state.explicit_fold(trans_trans_A, k, factor=5)
    m1, m2, k1, k2 = layer_state[trans_trans_A].axis()
    trans4_A = layer_state.explicit_shuffle(trans_trans_A, m1, k1, m2, k2)
    
    m, n = layer_state[E].axis()
    trans_E = layer_state.explicit_fold(E, m, factor=3)
    m1, m2, n = layer_state[E].axis()
    E_unfold = layer_state.explicit_unfold(E, m1, m2)

    input_A = ac.layer_tensor(layer_state[A].op.output(0).shape, name="real_A", dtype="float32")
    new_layer = layer_state.make_compute([input_A])
    out = new_layer.ops[0]
    sch = tvm.te.create_schedule(out)
    all_tensors = [*new_layer.inputs,
                   *new_layer.weights,
                   *new_layer.const_scalars,
                   *new_layer.const_tensors,
                   out.output(0)]
    print("New IR")
    print(tvm.lower(sch, all_tensors, simple_mode=True))
    new_func = tvm.build(sch, all_tensors, "llvm")
    M1, K1, M2, K2 = [int(x) for x in input_A.shape]
    new_A_np = np.random.uniform(-10, 10, [M1, K1, M2, K2]).astype("float32")
    for m1 in range(M1):
        for k1 in range(K1):
            for m2 in range(M2):
                for k2 in range(K2):
                    if m1 * 16 + m2 < M and k1 * 16 + k2 < K:
                        new_A_np[m1, k1, m2, k2] = A_np[m1 * 16 + m2, k1 * 16 + k2]
    new_F_np = np.random.uniform(-10, 10, [M, N]).astype("float32")
    new_A_tvm = tvm.nd.array(new_A_np, ctx)
    new_F_tvm = tvm.nd.array(new_F_np, ctx)
    new_func(new_A_tvm, B_tvm, C_tvm, D_tvm, 1.0, 1.0, new_F_tvm)

    from tvm import testing
    testing.assert_allclose(F_tvm.asnumpy(), new_F_tvm.asnumpy())
    print("Transform keeps the correctness!")


@register_test
def test3():
    pass


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
