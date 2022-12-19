import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
import argparse
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import pickle as pkl

EVALUTE_SCHEDULE_INPUTS = None
REPEAT = 2000

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def conv_relu_conv(
    N,
    C0,
    H,
    W,
    C1,
    R1,
    S1,
    C2,
    R2,
    S2,
    padding1=1,
    padding2=1,
    stride1=1,
    stride2=1,
    dtype="float32",
):
    P1 = (H + 2 * padding1 - R1) // stride1 + 1
    Q1 = (W + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1

    Img = tvm.te.placeholder([N, C0, H, W], dtype= dtype, name="Img")
    Weight1 = tvm.te.placeholder([C1, C0, R1, S1], dtype=dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([C2, C1, R2, S2], dtype=dtype, name="Weight2")

    # shared scope end

    rc1 = tvm.te.reduce_axis([0, C0], "rc1o")
    rr1 = tvm.te.reduce_axis([0, R1], "rr1")
    rs1 = tvm.te.reduce_axis([0, S1], "rs1")
    conv1 = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c1, p1, q1: tvm.te.sum(
            Img[n, rc1, p1 + rr1, q1 + rs1].astype(dtype)
            * Weight1[c1, rc1, rr1, rs1].astype(dtype),
            axis=[rc1, rr1, rs1],
        ),
        name="conv1",
    )
    relu1 = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c1, p1, q1: tvm.tir.if_then_else(
            conv1[n, c1, p1, q1] > tvm.tir.const(0, dtype),
            conv1[n, c1, p1, q1].astype(dtype),
            tvm.tir.const(0, dtype),
        ),
        name="relu1",
    )

    rc2 = tvm.te.reduce_axis([0, C1], "rc1o")
    rr2 = tvm.te.reduce_axis([0, R2], "rr1")
    rs2 = tvm.te.reduce_axis([0, S2], "rs1")
    conv2 = tvm.te.compute(
        [N, C2, P2, Q2],
        lambda n, c2, p2, q2: tvm.te.sum(
            relu1[n, rc2, p2 + rr2, q2 + rs2].astype(dtype)
            * Weight2[c2, rc2, rr2, rs2].astype(dtype),
            axis=[rc2, rr2, rs2],
        ),
        name="conv2",
    )

    
    return [Img, Weight1, Weight2, conv2]

def main(shape, dtype):
    target = tvm.target.Target(f"llvm -mcpu=skylake-avx512")
    task = tvm.auto_scheduler.SearchTask(func=conv_relu_conv, args=(*shape, dtype), target=target)

    # # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    os.system('mkdir -p logs')
    log_file = f"logs/conv_relu_conv_{shape}-{dtype}.json"

    n_line = 0
    if os.path.isfile(log_file):
        with open(log_file, 'rb') as f:
            n_line = len(f.readlines())
    n_line = min(n_line, 999)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000 - n_line,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    
    A, B, C, out = args

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    [Img, Weight1, Weight2, conv2] = conv_relu_conv(*shape, dtype)
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=[int(i) for i in Img.shape]).astype(dtype)
    b_np = np.random.uniform(size=[int(i) for i in Weight1.shape]).astype(dtype)
    c_np = np.random.uniform(size=[int(i) for i in Weight2.shape]).astype(dtype)
    out_np = np.random.uniform(size=[int(i) for i in conv2.shape]).astype(dtype)
    dev = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)
    evaluator = func.time_evaluator(
        func.entry_name, dev, min_repeat_ms=0, repeat=REPEAT, number = 1, f_preproc="cache_flush_cpu_non_first_arg"
    )
    cost = evaluator(a_tvm, b_tvm, c_tvm, out_tvm).mean
    N,C0,H,W,C1,R1,S1,C2,R2,S2,pad1,pad2,stride1,stride2 = shape
    workload = N * (C0 * C1 * H * W * R1 * S1 + C1 * C2 * R2 * S2 * H * W)
    topeak = workload / 1e9 / cost / 2995.2
    ret = {'time': cost, 'toPeak': topeak}
    print('shape, res')
    print(shape, ret)
    return ret


example_text = """
 example:
    python conv_relu_conv_ansor.py --dtype float32 --begin 0 --num 1
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
        with open("conv_relu_conv_ansor.pkl", 'wb') as f:
            pkl.dump(costs, f)
    print("shape,dtype,cost")
    for cc in costs:
        print(f"{cc[0], args.dtype, cc[1]}")
    with open(f"{args.output}/conv_relu_conv-ansor.pkl", 'wb') as f:
        pkl.dump(costs, f)
        