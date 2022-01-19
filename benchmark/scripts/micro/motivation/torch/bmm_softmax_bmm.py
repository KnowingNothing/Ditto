import torch
import numpy as np
import argparse
import time


def torch_bmm_softmax_bmm_cuda(A, B, C):
    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True
    with torch.autocast("cuda"):
        D = torch.bmm(A, B)
        E = torch.softmax(D, dim=-1)
        F = torch.bmm(E, C)

    return F


def main(profile):
    in_dtype = "float16"
    acc_dtype = "float32"

    repeat = 600
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[12, 512, 64], [12, 64, 512], [12, 512, 64]]
    ]

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    if profile:
        output = torch_bmm_softmax_bmm_cuda(*inputs_torch)
    else:
        # measure time
        # warm up
        output = torch_bmm_softmax_bmm_cuda(*inputs_torch)
        if not profile:
            records = []
            for i in range(repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                # beg = time.time()
                start.record()
                output = torch_bmm_softmax_bmm_cuda(*inputs_torch)
                assert output is not None
                end.record()
                torch.cuda.synchronize()
                # stop = time.time()
                total = start.elapsed_time(end)
                records.append(total)

            cost = np.mean(records)
            print(f"Average time cost is {cost} ms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    main(args.profile)
