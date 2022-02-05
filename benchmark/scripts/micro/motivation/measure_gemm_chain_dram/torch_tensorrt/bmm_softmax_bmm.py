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


def main(profile=False):
    model = Attention().eval()  # torch module needs to be in eval (not training) mode

    # here we use float32 to pass trace module
    # input_data_np = [
    #     np.random.uniform(-10, 10, [12, 512, 64]).astype("float32"),
    #     np.random.uniform(-10, 10, [12, 64, 512]).astype("float32"),
    #     np.random.uniform(-10, 10, [12, 512, 64]).astype("float32"),
    # ]
    # input_data_torch = [torch.tensor(x) for x in input_data_np]
    # traced_model = torch.jit.trace(model, input_data_torch)

    inputs = [
        torch_tensorrt.Input(
            [12, 512, 64],
            dtype=torch.half,
        ),
        torch_tensorrt.Input(
            [12, 64, 512],
            dtype=torch.half,
        ),
        torch_tensorrt.Input(
            [12, 512, 64],
            dtype=torch.half,
        ),
    ]
    enabled_precisions = {torch.float, torch.half}  # Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    repeat = 100
    test_data_np = [
        [
            np.random.uniform(-10, 10, [12, 512, 64]).astype("float16"),
            np.random.uniform(-10, 10, [12, 64, 512]).astype("float16"),
            np.random.uniform(-10, 10, [12, 512, 64]).astype("float16"),
        ]
        for x in range(repeat)
    ]

    torch_tensorrt.set_device(0)
    test_data_torch = [[torch.tensor(x).to("cuda") for x in y] for y in test_data_np]
    if profile:
        # only run once
        result = trt_ts_module(*test_data_torch[0])
    else:
        # measure time
        # warm up
        result = trt_ts_module(*test_data_torch[0])
        if not profile:
            records = []
            for i in range(repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                result = trt_ts_module(*test_data_torch[0])
                end.record()
                torch.cuda.synchronize()
                total = start.elapsed_time(end)
                records.append(total)

            cost = np.mean(records)
            print(f"Average time cost is {cost} ms.")
            # torch.jit.save(trt_ts_module, "trt_ts_module.ts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    main(args.profile)
