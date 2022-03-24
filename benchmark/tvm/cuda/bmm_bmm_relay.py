import tvm
from tvm import relay
import numpy as np
import argparse
import time


def relay_bmm_bmm(
    batch, M, N, K, L, in_dtype="float16", acc_dtype="float32", target="cuda"
):
    A = relay.var("A", shape=[batch, M, K], dtype=in_dtype)
    B = relay.var("B", shape=[batch, K, L], dtype=in_dtype)
    C = relay.var("C", shape=[batch, L, N], dtype=in_dtype)
    B = relay.transpose(B, axes=(0, 2, 1))
    D = relay.nn.batch_matmul(A, B, acc_dtype)
    E = relay.cast(D, dtype=in_dtype)
    C = relay.transpose(C, axes=(0, 2, 1))
    F = relay.nn.batch_matmul(E, C, acc_dtype)
    G = relay.cast(F, dtype=in_dtype)
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


def main(B, M, N, K, L, in_dtype, acc_dtype, only_once):
    in_dtype = in_dtype
    acc_dtype = acc_dtype
    target = "cuda -libs=cublas,cudnn"
    ins, outs, module = relay_bmm_bmm(
        B, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype, target=target
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in outs
    ]
    ctx = tvm.cuda()
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
    return cost


example_text = """
 example:
    python bmm_bmm_cuda.py --in_dtype float16 --acc_dtype float32 --begin 0 --num 1
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
        "--in_dtype",
        type=str,
        choices=["float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--acc_dtype",
        type=str,
        choices=["float16", "float32", "int32"],
        default="float16",
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
        cost = main(B, M, N, K, L, args.in_dtype, args.acc_dtype, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,L,in_dtype,acc_dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.in_dtype},{args.acc_dtype},{cc[1]}")
