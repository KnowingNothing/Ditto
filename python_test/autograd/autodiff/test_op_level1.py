import tvm
import torch
import numpy as np
from tvm import testing
from tvm import auto_scheduler as ansor
from ditto import autograd as ag

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
    Ashape = [10, 20, 1152]
    Cshape = [10, 20, 1152, 16]
    dtype = "float32"

    A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")
    B = tvm.te.placeholder(Cshape, dtype=dtype, name="B")

    C = tvm.te.compute(Cshape,
    lambda i, j, k, n: B[i,j,k,n]+A[i,j,k], name="C")

    dC = tvm.te.placeholder(Cshape, dtype=dtype, name="dC")

    dA = ag.grad_op(A, C, dC)

    s = tvm.te.create_schedule(dA.op)

    print(tvm.lower(s, [A, B, dC, dA], simple_mode=True))
    func = tvm.build(s, [A, B, dC, dA], target="llvm")

    A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
    B_np = np.random.uniform(-10, 10, Cshape).astype("float32")
    dC_np = np.ones(Cshape).astype("float32")
    dA_np = np.zeros(Ashape).astype("float32")

    ctx = tvm.cpu(0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    dC_tvm = tvm.nd.array(dC_np, ctx)
    dA_tvm = tvm.nd.array(dA_np, ctx)

    func(A_tvm, B_tvm, dC_tvm, dA_tvm)

    print(dA_tvm)

    # =======>
    # compare the results with Pytorch
    A_torch = torch.tensor(A_np, requires_grad=True)
    B_torch = torch.tensor(B_np, requires_grad=True)
    C_torch = B_torch + A_torch.reshape(Ashape + [1]).expand_as(B_torch)
    loss = C_torch.sum()
    loss.backward()
    testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
    print("Compare with Numpy success!")
    
    
@register_test
def test2():
    N = 2
    nC = 16
    H = 14
    W = 14
    K = 8
    R = 3
    S = 3

    st = 1

    P = (H - R + 1) // st
    Q = (W - S + 1) // st

    dtype = "float32"

    A = tvm.te.placeholder([N, nC, H, W], dtype=dtype, name="A")
    B = tvm.te.placeholder([K, nC, R, S], dtype=dtype, name="B")
    c = tvm.te.reduce_axis([0, nC], name="c")
    r = tvm.te.reduce_axis([0, R], name="r")
    s = tvm.te.reduce_axis([0, S], name="s")
    C = tvm.te.compute([N, K, P, Q],
    lambda n, k, h, w :
        tvm.te.sum(A[n, c, h * st + r, w * st + s] * B[k, c, r, s], axis=[c,r,s]), name="C")

    dC = tvm.te.placeholder([N, K, P, Q], dtype=dtype, name="dC")

    print(C.op.body)

    # print(dir(C.op.body[0].source[0]))

    # print(tvm.tg.expr_equal(C.op.body[0].source[0].b.args[0], C.op.body[0].source[0].b.args[1]))

    dA = ag.grad_op(A, C, dC)

    s = tvm.te.create_schedule(dA.op)

    print(tvm.lower(s, [B, dC, dA], simple_mode=True))

    func = tvm.build(s, [B, dC, dA], target="llvm")

    A_np = np.random.uniform(-10, 10, [N, nC, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, nC, R, S]).astype("float32")
    dC_np = np.random.uniform(-10, 10, [N, K, P, Q]).astype("float32")
    dA_np = np.zeros([N, nC, H, W]).astype("float32")

    ctx = tvm.cpu(0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    dC_tvm = tvm.nd.array(dC_np, ctx)
    dA_tvm = tvm.nd.array(dA_np, ctx)

    func(B_tvm, dC_tvm, dA_tvm)

    print(dA_tvm)

    # =======>
    # compare the results with pytorch
    A_torch = torch.tensor(A_np)
    B_torch = torch.tensor(B_np)
    dC_torch = torch.tensor(dC_np)
    golden_torch = torch.nn.functional.conv_transpose2d(dC_torch, B_torch)
    testing.assert_allclose(dA_tvm.asnumpy(), golden_torch.numpy(), rtol=1e-3, atol=1e-3)
    print("Compare with PyTorch success!")
    
    
@register_test
def test3():
    N = 2
    nC = 16
    H = 14
    W = 14
    K = 8
    R = 3
    S = 3

    st = 1

    P = (H - R + 1) // st
    Q = (W - S + 1) // st

    dtype = "float32"

    A = tvm.te.placeholder([N, nC, H, W], dtype=dtype, name="A")
    B1 = tvm.te.placeholder([K, nC, R, S], dtype=dtype, name="B1")
    
    B = tvm.te.compute([K, nC, R, S], lambda k, c, r, s: B1[k, c, r, s] * B1[k, c, r, s], name="B")
    c = tvm.te.reduce_axis([0, nC], name="c")
    r = tvm.te.reduce_axis([0, R], name="r")
    s = tvm.te.reduce_axis([0, S], name="s")
    C = tvm.te.compute([N, K, P, Q],
    lambda n, k, h, w :
        tvm.te.sum(A[n, c, h * st + r, w * st + s] * B[k, c, r, s], axis=[c,r,s]), name="C")

    dC = tvm.te.placeholder([N, K, P, Q], dtype=dtype, name="dC")

    print(C.op.body)

    # print(dir(C.op.body[0].source[0]))

    # print(tvm.tg.expr_equal(C.op.body[0].source[0].b.args[0], C.op.body[0].source[0].b.args[1]))

    dA = ag.grad_op(A, C, dC)

    s = tvm.te.create_schedule(dA.op)

    print(tvm.lower(s, [B1, dC, dA], simple_mode=True))

    func = tvm.build(s, [B1, dC, dA], target="llvm")

    A_np = np.random.uniform(-10, 10, [N, nC, H, W]).astype("float32")
    B1_np = np.random.uniform(-10, 10, [K, nC, R, S]).astype("float32")
    dC_np = np.random.uniform(-10, 10, [N, K, P, Q]).astype("float32")
    dA_np = np.zeros([N, nC, H, W]).astype("float32")

    ctx = tvm.cpu(0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B1_tvm = tvm.nd.array(B1_np, ctx)
    dC_tvm = tvm.nd.array(dC_np, ctx)
    dA_tvm = tvm.nd.array(dA_np, ctx)

    func(B1_tvm, dC_tvm, dA_tvm)

    print(dA_tvm)

    # =======>
    # compare the results with pytorch
    A_torch = torch.tensor(A_np)
    B1_torch = torch.tensor(B1_np)
    B_torch = B1_torch * B1_torch
    dC_torch = torch.tensor(dC_np)
    golden_torch = torch.nn.functional.conv_transpose2d(dC_torch, B_torch)
    testing.assert_allclose(dA_tvm.asnumpy(), golden_torch.numpy(), rtol=1e-3, atol=1e-3)
    print("Compare with PyTorch success!")


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
