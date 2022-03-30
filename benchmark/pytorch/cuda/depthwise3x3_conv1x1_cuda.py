import torch
import numpy as np
import argparse
import time


def torch_depthwise3x3_conv1x1_cuda(A, B, C, groups, stride1, stride2):
    with torch.autocast("cuda"):
        D = torch.nn.functional.conv2d(A, B, stride=stride1, groups=groups, padding=1)
        F = torch.nn.functional.conv2d(D, C, stride=stride2, padding=0)

    return F


def main(batch, C, H, W, K, stride1, stride2, factor, dtype, only_once=False):
    in_dtype = dtype

    repeat = 600
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[batch, C, H, W], [C * factor, 1, 3, 3], [K, C * factor, 1, 1]]
    ]

    if only_once:
        inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
        output = torch_depthwise3x3_conv1x1_cuda(*inputs_torch, C, stride1, stride2)
        cost = -1
    else:
        inputs_torch = [
            [torch.tensor(x).cuda() for x in inputs_np] for i in range(repeat)
        ]
        # measure time
        # warm up
        for i in range(repeat):
            output = torch_depthwise3x3_conv1x1_cuda(*inputs_torch[i], C, stride1, stride2)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # beg = time.time()
        start.record()
        for i in range(repeat):
            output = torch_depthwise3x3_conv1x1_cuda(*inputs_torch[i], C, stride1, stride2)
        end.record()
        torch.cuda.synchronize()
        # stop = time.time()
        total = start.elapsed_time(end)
        cost = total / repeat
        # print(f"Average time cost is {cost} ms.")
    return cost


example_text = """
 example:
    python depthwise3x3_conv1x1_cuda.py --dtype float16 --begin 0 --num 1
"""


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # (C, H, W, K, stride1, stride2)
    # # mobilenet-v1
    # (32, 112, 112, 64, 1, 1),
    # (64, 112, 112, 128, 2, 1),
    # (128, 56, 56, 128, 1, 1),
    # (128, 56, 56, 256, 2, 1),
    # (256, 28, 28, 256, 1, 1),
    # (256, 28, 28, 512, 2, 1),
    # (512, 14, 14, 512, 1, 1),
    # (512, 14, 14, 1024, 2, 1),
    # small channel
    (32, 112, 112, 64, 1, 1),  # mobilenet-v1
    (32, 56, 56, 64, 1, 1),  # dummy
    (64, 112, 112, 64, 1, 1),  # dummy
    (16, 224, 224, 32, 2, 1),  # dummy
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--batch", type=int, default=1
    )
    parser.add_argument(
        "--factor", type=int, default=1
    )

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        C, H, W, K, stride1, stride2 = ss
        cost = main(args.batch, C, H, W, K, stride1, stride2, args.factor, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("batch,C,H,W,K,stride1,stride2,factor,dtype,cost")
    for cc in costs:
        print(
            f"{args.batch}," + ",".join(map(str, cc[0])) + f",{args.factor},{args.dtype},{cc[1]}"
        )
