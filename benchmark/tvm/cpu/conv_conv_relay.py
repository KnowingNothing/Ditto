import tvm
from tvm import relay
import numpy as np
import argparse
import time
import pickle as pkl
import os

def relay_conv_conv(
    shape, dtype="float32", target="llvm"
):
    N, C0, H, W, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape 
    A = relay.var("Img", shape=[N, C0, H, W], dtype=dtype)
    B = relay.var("Weight1", shape=[C1, C0, R1, S1], dtype=dtype)
    C = relay.var("Weight2", shape=[C2, C1, R2, S2], dtype=dtype)
    D = relay.nn.conv2d(A, B, strides=(stride1, stride1), padding=(padding1, padding1), dilation=(1, 1), groups=1, channels=C1, kernel_size=(R1,S1), data_layout='NCHW', kernel_layout='OIHW', out_layout='', out_dtype='')
    F = relay.nn.conv2d(D, C, strides=(stride1, stride1), padding=(padding1, padding1), dilation=(1, 1), groups=1, channels=C2, kernel_size=(R2, S2), data_layout='NCHW', kernel_layout='OIHW', out_layout='', out_dtype='')
    args = relay.analysis.free_vars(F)
    print("args: ", args)
    func = relay.Function(args, F)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    return [[N, C0, H, W], [C1, C0, R1, S1], [C2, C1, R2, S2]], [[N, C2, H, W]], module


def main(shape, dtype):
    target = "llvm -mcpu=skylake-avx512"
    ins, outs, module = relay_conv_conv(
        shape, dtype, target
    )
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype) for y in outs
    ]
    dev = tvm.device(str(target), 0)
    ret = module.benchmark(dev, number = 1000, repeat = 1)
    cost = ret.mean
    N,C0,H,W,C1,R1,S1,C2,R2,S2,pad1,pad2,stride1,stride2 = shape
    workload = N * (C0 * C1 * H * W * R1 * S1 + C1 * C2 * R2 * S2 * H * W)
    topeak = workload / 1e9 / cost / 2995.2
    ret = {'time': cost, 'toPeak': topeak}
    print('shape, res')
    print(shape, ret)
    return ret


example_text = """
 example:
    python conv_conv_relay.py --dtype float32 --begin 0 --num 1
"""

shapes = [
    [1, 64, 114, 112, 192, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 32, 147, 148, 64, 3, 3, 96, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 128, 3, 3, 64, 1, 1, 1, 0, 1, 1],
    [1, 128, 27, 28, 256, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 16, 228, 227, 64, 3, 3, 32, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 64, 1, 1, 64, 3, 3, 0, 1, 1, 1],
    [1, 64, 57, 56, 64, 1, 1, 64, 1, 1, 0, 0, 1, 1],
    [1, 256, 57, 56, 256, 1, 1, 64, 1, 1, 0, 0, 1, 1]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--output", type = str, default = "result"
    )

    args = parser.parse_args()
    
    os.system(f'mkdir -p {args.output}')
    
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        cost = main(ss, args.dtype)
        costs.append((ss, cost))
    print("shape,dtype,cost")
    for cc in costs:
        print(f"{cc[0], args.dtype, cc[1]}")
    with open(f"{args.output}/conv_conv-relay.pkl", 'wb') as f:
        pkl.dump(costs, f)