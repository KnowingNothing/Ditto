import tvm
from tvm import relay
import numpy as np
import argparse 
import time
from tvm import topi
from tvm import te
import os
import pickle as pkl

REPEAT = 2000
peakflops = {'sc': 704, 'sccc': 2150, 'scccc': 2995}
def relay_bmm_bmm(
    batch, M, N, K, L, dtype="float32", target="cpu"
):
    A = relay.var("A", shape=[batch, M, K], dtype=dtype)
    B = relay.var("B", shape=[batch, K, L], dtype=dtype)
    C = relay.var("C", shape=[batch, L, N], dtype=dtype)
    D = relay.nn.batch_matmul(A, B, dtype, transpose_b = False)
    E = relay.nn.batch_matmul(D, C, dtype, transpose_b = False)
    args = relay.analysis.free_vars(E)
    func = relay.Function(args, E)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    t0 = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    t1 = time.time()
    return [[batch, M, K], [batch, K, L], [batch, L, N]], [[batch, M, N]], module, t1 - t0



def main(B, M, N, K, L, dtype, server):
    target = "llvm -mcpu=skylake-avx512"
    ins, outs, module, search_time = relay_bmm_bmm(
        B, M, N, K, L, dtype=dtype, target=target
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in outs
    ]
    dev = tvm.device(str(target), 0)
    ret = module.benchmark(dev, number = REPEAT, repeat = 1)
    cost = ret.mean
    workload = B * (M * K * L + M * N * L)
    ratioToPeak = workload / 1e9 / cost / peakflops[server]
    return (cost, ratioToPeak, search_time)




example_text = """
 example:
    python bmm_bmm.py --dtype float32 --begin 0 --num 1
"""

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512), # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),   # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256), # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256), # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),   # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)), # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)), # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512), # Mixer-Large/32-S
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
    parser.add_argument(
        "--server", type=str, choices=peakflops.keys()
    )

    args = parser.parse_args()
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(B, M, N, K, L, dtype = args.dtype, server = args.server)
        costs.append((ss, cost))
    print("B,M,N,K,dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{args.dtype},{args.server},{cc[1]}")
    with open("bmm_bmm_relay.pkl", 'wb') as f:
        pkl.dump(costs, f)