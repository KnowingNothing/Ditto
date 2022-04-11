import torch
import numpy as np
import argparse
import time


def torch_conv3x3_conv1x1_cuda(A, B, C, stride1, stride2):
    with torch.autocast("cuda"):
        D = torch.nn.functional.conv2d(A, B, stride=stride1, padding=1)
        F = torch.nn.functional.conv2d(D, C, stride=stride2, padding=0)

    return F


def main(batch, C, H, W, K1, K2, stride1, stride2, dtype, only_once=False):
    in_dtype = dtype

    repeat = 600
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[batch, C, H, W], [K1, C, 3, 3], [K2, K1, 1, 1]]
    ]

    if only_once:
        inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
        output = torch_conv3x3_conv1x1_cuda(*inputs_torch, stride1, stride2)
        cost = -1
    else:
        inputs_torch = [
            [torch.tensor(x).cuda() for x in inputs_np] for i in range(repeat)
        ]
        # measure time
        # warm up
        for i in range(repeat):
            output = torch_conv3x3_conv1x1_cuda(*inputs_torch[i], stride1, stride2)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # beg = time.time()
        start.record()
        for i in range(repeat):
            output = torch_conv3x3_conv1x1_cuda(*inputs_torch[i], stride1, stride2)
        end.record()
        torch.cuda.synchronize()
        # stop = time.time()
        total = start.elapsed_time(end)
        cost = total / repeat
        # print(f"Average time cost is {cost} ms.")
    return cost


example_text = """
 example:
    python conv3x3_conv1x1_cuda.py --dtype float16 --begin 0 --num 1
"""


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    (64, 112, 112, 192, 128, 2, 1),  # Yolo
    (32, 147, 147, 64, 80, 2, 1), # Inception-V3
    (64, 56, 56, 128, 64, 1, 1), # Darknet-19
    (128, 28, 28, 256, 128, 1, 1), # Darknet-19
    (16, 227, 227, 64, 16, 4, 1), # Squeezenet-V1.1
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

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        C, H, W, K1, K2, stride1, stride2 = ss
        cost = main(args.batch, C, H, W, K1, K2, stride1, stride2, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("batch,C,H,W,K1,K2,stride1,stride2,dtype,cost")
    for cc in costs:
        print(
            f"{args.batch}," + ",".join(map(str, cc[0])) + f",{args.dtype},{cc[1]}"
        )
