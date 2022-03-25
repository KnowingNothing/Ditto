import tvm
from tvm import relay
import numpy as np
import argparse
import time
from tvm import topi
from tvm import te
import os

def relay_bmm_bmm(
    batch, M, N, K, L, dtype="float32", target="cpu"
):
    A = relay.var("A", shape=[batch, M, K], dtype=dtype)
    B = relay.var("B", shape=[batch, K, L], dtype=dtype)
    C = relay.var("C", shape=[batch, L, N], dtype=dtype)
    B = relay.transpose(B, axes=(0, 2, 1))
    D = relay.nn.batch_matmul(A, B, dtype)
    E = relay.cast(D, dtype=dtype)
    C = relay.transpose(C, axes=(0, 2, 1))
    F = relay.nn.batch_matmul(E, C, dtype)
    G = relay.cast(F, dtype=dtype)
    args = relay.analysis.free_vars(G)
    func = relay.Function(args, G)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    return [[batch, M, K], [batch, K, L], [batch, L, N]], [[batch, M, N]], module


def main(B, M, N, K, L, dtype, only_once):
    target = "llvm -mcpu=core-avx2"
    ins, outs, module = test_time(
        B, M, N, K, L, dtype=dtype, target=target
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in outs
    ]
    ctx = tvm.cpu()
    dev = tvm.device(str(target), 0)
    if only_once:
        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
        module.set_input(key=0, value=inputs_tvm[0])
        module.set_input(key=1, value=inputs_tvm[1])
        module.set_input(key=2, value=inputs_tvm[2])
        # module.set_input(key=3, value=outputs_tvm[0])
        module.run()
        cost = -1
    else:
        ret = module.benchmark(dev, min_repeat_ms=600)
        cost = ret.mean * 1e3
    workload = B * (M * K * L + M * N * L)
    ratioToPeak = workload / (cost * 2.2 * 1e6 * 20 * 35.2)
    return cost, ratioToPeak

def test_time(batch, M, N, K, L, dtype = "float32", target = "llvm -mcpu=core-avx2"):
    os.environ["TVM_NUM_THREADS"] = str(20)
    with tvm.target.Target(target) as tgt:
        dev = tvm.device(tgt.kind.name, 0)

        Qtensor = tvm.te.placeholder([batch, M, K], name="Q", dtype=dtype)
        Ktensor = tvm.te.placeholder([batch, K, L], name="K", dtype=dtype)

        QK = topi.x86.batch_matmul(Qtensor, Ktensor, None, None, False, False)
        s = topi.x86.schedule_batch_matmul([QK])

        f = tvm.build(s, [Qtensor, Ktensor, QK], target=tgt)

        a = tvm.nd.array(np.random.uniform(size=[batch, M, K]).astype(Qtensor.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=[batch, K, L]).astype(Ktensor.dtype), dev)
        c = tvm.nd.array(np.zeros([batch, M, L], dtype=QK.dtype), dev)
        f(a, b, c)

        time_f = f.time_evaluator(f.entry_name, dev, number=10000)
        cost = time_f(a, b, c).mean

        QK = tvm.te.placeholder([batch, M, L], name="QK", dtype=dtype)
        QK_softmax = topi.nn.softmax(QK)
        s = topi.x86.schedule_softmax([QK_softmax])

        f = tvm.build(s, [QK, QK_softmax], target=tgt)

        a = tvm.nd.array(np.random.uniform(size=[batch, M, L]).astype(QK.dtype), dev)
        c = tvm.nd.array(np.zeros([batch, M, L], dtype=QK_softmax.dtype), dev)
        f(a, c)

        time_f = f.time_evaluator(f.entry_name, dev, number=10)
        cost = cost + time_f(a, c).mean

        QK_softmax = tvm.te.placeholder([batch, M, L], name="QK_softmax", dtype=dtype)
        Vtensor = tvm.te.placeholder([batch, L, N], name="K", dtype=dtype)

        QKV = topi.x86.batch_matmul(QK_softmax, Vtensor, None, None, False, False)
        s = topi.x86.schedule_batch_matmul([QKV])

        f = tvm.build(s, [QK_softmax, Vtensor, QKV], target=tgt)

        a = tvm.nd.array(np.random.uniform(size=[batch, M, L]).astype(Qtensor.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=[batch, L, N]).astype(Ktensor.dtype), dev)
        c = tvm.nd.array(np.zeros([batch, M, N], dtype=QK.dtype), dev)
        f(a, b, c)

        time_f = f.time_evaluator(f.entry_name, dev, number=10000)
        cost = cost + time_f(a, b, c).mean
    workload = B * (M * K * L + M * N * L)
    ratioToPeak = workload / (cost * 2.2 * 1e9 * 20 * 35.2)
    return cost, ratioToPeak


example_text = """
 example:
    python bmm_bmm.py --dtype float32 --begin 0 --num 1
"""

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32"],
        default="float32",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = test_time(B, M, N, K, L, args.dtype)
        costs.append((ss, cost))
    print("B,M,N,K,dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{args.dtype},{cc[1]}")
