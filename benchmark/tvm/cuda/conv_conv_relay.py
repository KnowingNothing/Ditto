import tvm
from tvm import relay
import numpy as np
import argparse
import time


def relay_conv_conv(
    batch,
    C,
    H,
    W,
    K1,
    K2,
    stride1,
    stride2,
    k1,
    k2,
    in_dtype="float16",
    acc_dtype="float32",
    target="cuda",
):
    Img = relay.var("Img", shape=[batch, C, H, W], dtype=in_dtype)
    W1 = relay.var("W1", shape=[K1, C, k1, k1], dtype=in_dtype)
    W2 = relay.var("W2", shape=[K2, K1, k2, k2], dtype=in_dtype)
    Conv1 = relay.nn.conv2d(
        Img, W1, strides=(stride1, stride1), padding=(k1 // 2, k1 // 2),
        out_dtype=acc_dtype
    )
    cast = relay.cast(Conv1, dtype=in_dtype)
    Conv2 = relay.nn.conv2d(
        cast, W2, strides=(stride2, stride2), padding=(k2 // 2, k2 // 2),
        out_dtype=acc_dtype
    )
    cast = relay.cast(Conv2, dtype=in_dtype)
    args = relay.analysis.free_vars(cast)
    func = relay.Function(args, cast)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    return (
        [[batch, C, H, W], [K1, C, k1, k1], [K2, K1, k2, k2]],
        [[batch, K2, H // stride1 // stride2, W // stride1 // stride2]],
        module,
    )


def main(
    batch, C, H, W, K1, K2, stride1, stride2, k1, k2, in_dtype, acc_dtype, only_once
):
    in_dtype = in_dtype
    acc_dtype = acc_dtype
    target = "cuda -libs=cublas,cudnn"
    ins, outs, module = relay_conv_conv(
        batch,
        C,
        H,
        W,
        K1,
        K2,
        stride1,
        stride2,
        k1,
        k2,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
        target=target,
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
    python conv_conv_relay.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
"""


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # C, H, W, K1, K2, st1, st2, k1, k2
    (64, 112, 112, 192, 128, 2, 1, 3, 1),  # Yolo
    (32, 147, 147, 64, 80, 2, 1, 3, 1),  # Inception-V3
    (64, 56, 56, 128, 64, 1, 1, 3, 1),  # Darknet-19
    (128, 28, 28, 256, 128, 1, 1, 3, 1),  # Darknet-19
    (16, 227, 227, 64, 16, 4, 1, 3, 1),  # Squeezenet-V1.1
    (64, 56, 56, 64, 64, 1, 1, 1, 3),  # ResNet-50
    (64, 56, 56, 64, 64, 1, 1, 1, 1),  # modified ResNet-50
    (256, 56, 56, 256, 64, 1, 1, 1, 1),  # modified ResNet-50
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
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        C, H, W, K1, K2, stride1, stride2, k1, k2 = ss
        cost = main(
            args.batch,
            C,
            H,
            W,
            K1,
            K2,
            stride1,
            stride2,
            k1,
            k2,
            args.in_dtype,
            args.acc_dtype,
            args.only_once,
        )
        costs.append((ss, cost))
    print("batch,C,H,W,K1,K2,stride1,stride2,k1,k2,in_dtype,acc_dtype,cost")
    for cc in costs:
        print(f"{args.batch}," + ",".join(map(str, cc[0])) + f",{args.in_dtype},{args.acc_dtype},{cc[1]}")
