import torch
import numpy as np
import argparse
import time


def torch_bmm_softmax_bmm_cuda(A, B, C):
    with torch.autocast("cuda"):
        D = torch.bmm(A, B)
        E = torch.softmax(D, dim=-1)
        F = torch.bmm(E, C)

    return F


def main(B, M, N, K, L, dtype, only_once=False):
    in_dtype = dtype

    repeat = 600
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[B, M, K], [B, K, L], [B, L, N]]
    ]

    if only_once:
        inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
        output = torch_bmm_softmax_bmm_cuda(*inputs_torch)
        cost = -1
    else:
        inputs_torch = [
            [torch.tensor(x).cuda() for x in inputs_np] for i in range(repeat)
        ]
        # measure time
        # warm up
        for i in range(repeat):
            output = torch_bmm_softmax_bmm_cuda(*inputs_torch[i])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # beg = time.time()
        start.record()
        for i in range(repeat):
            output = torch_bmm_softmax_bmm_cuda(*inputs_torch[i])
        end.record()
        torch.cuda.synchronize()
        # stop = time.time()
        total = start.elapsed_time(end)
        cost = total / repeat
        # print(f"Average time cost is {cost} ms.")
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

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(B, M, N, K, L, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{args.dtype},{cc[1]}")
