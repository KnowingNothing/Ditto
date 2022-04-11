import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import argparse
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


EVALUTE_SCHEDULE_INPUTS = None


def evaluate_schedule_worker(dummy):
    global EVALUTE_SCHEDULE_INPUTS
    sch, args, ins, outs, sm = EVALUTE_SCHEDULE_INPUTS
    func = tvm.build(sch, args, f"cuda -arch=sm_{sm}")
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in outs
    ]
    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    # print(f"Our code uses {cost} ms")
    return cost


def evaluate_schedule(sch, args, ins, outs, sm):
    global EVALUTE_SCHEDULE_INPUTS
    EVALUTE_SCHEDULE_INPUTS = (sch, args, ins, outs, sm)
    with ProcessPool(1) as pool:
        future = pool.map(evaluate_schedule_worker, [0], timeout=100)
        iterator = future.result()

        while True:
            try:
                results = next(iterator)
                # print(".Y", end="", flush=True)
            except StopIteration:
                break
            except TimeoutError as error:
                print(".T", end="", flush=True)
                results = 1e10
            except Exception as error:
                print(error)
                print(".E", end="", flush=True)
                results = 1e10

        return results


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def conv_conv(batch, C, H, W, K1, K2, stride1, stride2, k1, k2, in_dtype, acc_dtype):
    Img = te.placeholder((batch, C, H, W), name="Img", dtype=in_dtype)
    W1 = te.placeholder((K1, C, k1, k1), name="W1", dtype=in_dtype)
    W2 = te.placeholder((K2, K1, k2, k2), name="W2", dtype=in_dtype)

    conv1 = topi.nn.conv2d_nchw(
        Img, W1, stride1, k1 // 2, out_dtype=acc_dtype, dilation=1
    )
    cast1 = tvm.te.compute(
        conv1.shape, lambda *args: conv1(*args).astype(in_dtype), name="cast"
    )
    relu = tvm.te.compute(
        conv1.shape,
        lambda *args: tvm.tir.if_then_else(
            cast1(*args) >= tvm.tir.const(0, in_dtype),
            cast1(*args),
            tvm.tir.const(0, in_dtype),
        ),
        name="relu",
    )
    conv2 = topi.nn.conv2d_nchw(
        relu, W2, stride2, k2 // 2, out_dtype=acc_dtype, dilation=1
    )
    cast2 = tvm.te.compute(
        conv2.shape, lambda *args: conv2(*args).astype(in_dtype), name="cast"
    )

    return [Img, W1, W2, cast2]


def main(
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
    in_dtype,
    acc_dtype,
    sm="70",
    only_once=False,
    test=False,
):
    target = tvm.target.Target(f"cuda -arch=sm_{sm}")
    task = tvm.auto_scheduler.SearchTask(
        func=conv_conv,
        args=(batch, C, H, W, K1, K2, stride1, stride2, k1, k2, in_dtype, acc_dtype),
        target=target,
    )

    # # Inspect the computational graph
    # print("Computational DAG:")
    # print(task.compute_dag)

    log_file = f"conv_relu_conv_{batch}-{C}-{H}-{W}-{K1}-{K2}-{stride1}-{stride2}-{k1}-{k2}-{in_dtype}-{acc_dtype}.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # Run auto-tuning (search)
    if not test:
        task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    A, B, C, out = args

    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))

    if only_once:
        func = tvm.build(sch, args, target)
        a_np = np.random.uniform(size=[int(x) for x in A.shape]).astype(in_dtype)
        b_np = np.random.uniform(size=[int(x) for x in B.shape]).astype(in_dtype)
        c_np = np.random.uniform(size=[int(x) for x in C.shape]).astype(in_dtype)
        out_np = np.random.uniform(size=[int(x) for x in out.shape]).astype(in_dtype)
        dev = tvm.cuda()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        out_tvm = tvm.nd.array(out_np, device=dev)
        func(a_tvm, b_tvm, c_tvm, out_tvm)
        cost = -1
    else:
        # cost = evaluate_schedule(sch, args, [A, B, C], [out], sm)
        func = tvm.build(sch, args, target)
        a_np = np.random.uniform(size=[int(x) for x in A.shape]).astype(in_dtype)
        b_np = np.random.uniform(size=[int(x) for x in B.shape]).astype(in_dtype)
        c_np = np.random.uniform(size=[int(x) for x in C.shape]).astype(in_dtype)
        out_np = np.random.uniform(size=[int(x) for x in out.shape]).astype(in_dtype)
        dev = tvm.cuda()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        out_tvm = tvm.nd.array(out_np, device=dev)
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        cost = evaluator(a_tvm, b_tvm, c_tvm, out_tvm).mean * 1e3
    return cost


example_text = """
 example:
    python conv_relu_conv_ansor.py --in_dtype float16 --acc_dtype float32 --begin 0 --num 1 --sm 80
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
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "float64", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--acc_dtype",
        type=str,
        choices=["float16", "float32", "float64", "int32"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--sm",
        type=str,
        choices=["70", "80", "86"],
        default="70",
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
            in_dtype=args.in_dtype,
            acc_dtype=args.acc_dtype,
            sm=args.sm,
            only_once=args.only_once,
            test=args.test,
        )
        costs.append((ss, cost))

    print("batch,C,H,W,K1,K2,stride1,stride2,k1,k2,in_dtype,acc_dtype,cost")
    for cc in costs:
        print(
            f"{args.batch},"
            + ",".join(map(str, cc[0]))
            + f",{args.in_dtype},{args.acc_dtype},{cc[1]}"
        )
