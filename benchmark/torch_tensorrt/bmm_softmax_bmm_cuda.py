# To run this file
# 1. source the cuda env
# 2. source the tensorrt env
# 3. source the torch_tensorrt env
from ast import arg
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import torch_tensorrt
import argparse


# Network
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        x = torch.bmm(q, k)
        x = self.softmax(x)
        x = torch.bmm(x, v)
        return x


def main(B, M, N, K, L, dtype, only_once=False):
    model = Attention().eval()  # torch module needs to be in eval (not training) mode

    # here we use float32 to pass trace module
    # input_data_np = [
    #     np.random.uniform(-10, 10, [12, 512, 64]).astype("float32"),
    #     np.random.uniform(-10, 10, [12, 64, 512]).astype("float32"),
    #     np.random.uniform(-10, 10, [12, 512, 64]).astype("float32"),
    # ]
    # input_data_torch = [torch.tensor(x) for x in input_data_np]
    # traced_model = torch.jit.trace(model, input_data_torch)

    trt_dtype = torch.half
    if dtype == "float16":
        enabled_precisions = {torch.float, torch.half}  # Run with fp16
    elif dtype == "int8":
        trt_dtype = torch.int8
        enabled_precisions = {torch.int32, torch.int8}  # Run with int8
    else:
        raise RuntimeError(f"datatype {dtype} not supported.")

    inputs = [
        torch_tensorrt.Input(
            [B, M, K],
            dtype=trt_dtype,
        ),
        torch_tensorrt.Input(
            [B, K, L],
            dtype=trt_dtype,
        ),
        torch_tensorrt.Input(
            [B, L, N],
            dtype=trt_dtype,
        ),
    ]

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    repeat = 600
    test_data_np = [
        [
            np.random.uniform(-10, 10, [B, M, K]).astype(dtype),
            np.random.uniform(-10, 10, [B, K, L]).astype(dtype),
            np.random.uniform(-10, 10, [B, L, N]).astype(dtype),
        ]
        for x in range(repeat)
    ]

    torch_tensorrt.set_device(0)
    test_data_torch = [[torch.tensor(x).to("cuda") for x in y] for y in test_data_np]
    if only_once:
        # only run once
        result = trt_ts_module(*test_data_torch[0])
        cost = -1
    else:
        # measure time
        # warm up
        for i in range(repeat):
            result = trt_ts_module(*test_data_torch[i])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for i in range(repeat):
            result = trt_ts_module(*test_data_torch[i])
        end.record()
        torch.cuda.synchronize()
        total = start.elapsed_time(end)
        cost = total / repeat
        # print(f"Average time cost is {cost} ms.")
        # torch.jit.save(trt_ts_module, "trt_ts_module.ts")
    return cost


example_text = """
 example:
    python bmm_softmax_bmm_cuda.py --dtype float16 --begin 0 --num 1
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
        choices=["float16", "int8"],
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
        cost = main(B, M, N, K, L, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{args.dtype},{cc[1]}")
