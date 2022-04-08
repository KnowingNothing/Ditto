import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
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
def bmm_bmm(batch, M, N, K, L, in_dtype, acc_dtype):
    A = te.placeholder((batch, M, K), name="A", dtype=in_dtype)
    B = te.placeholder((batch, K, L), name="B", dtype=in_dtype)
    C = te.placeholder((batch, L, N), name="C", dtype=in_dtype)

    k = te.reduce_axis((0, K), name="k")
    D = te.compute(
        (batch, M, L),
        lambda b, m, l: te.sum(A[b, m, k].astype(acc_dtype) * B[b, k, l].astype(acc_dtype), axis=k),
        name="D"
    )
    # exp = te.compute(
    #     (B, M, L),
    #     lambda b, m, l: te.exp(D[b, m, l])
    # )
    # rl = te.reduce_axis((0, L), name="rl")
    # sumv = te.compute(
    #     (B, M),
    #     lambda b, m: te.sum(exp[b, m, rl], axis=rl)
    # )
    # softmax = te.compute(
    #     (B, M, L),
    #     lambda b, m, l: (exp[b, m, l] / sumv[b, m]).astype(in_dtype)
    # )
    rrl = te.reduce_axis((0, L), name="rrl")
    E = te.compute(
        (batch, M, N),
        lambda b, m, n: te.sum(D[b, m, rrl].astype(acc_dtype) * C[b, rrl, n].astype(acc_dtype), axis=rrl),
        name="E"
    )
    
    out = te.compute(
        (batch, M, N),
        lambda b, m, n: E[b, m, n].astype(in_dtype),
        name="out"
    )

    return [A, B, C, out]

def main(batch, M, N, K, L, in_dtype, acc_dtype, sm="70", only_once=False, test=False):
    target = tvm.target.Target(f"cuda -arch=sm_{sm}")
    task = tvm.auto_scheduler.SearchTask(func=bmm_bmm, args=(batch, M, N, K, L, in_dtype, acc_dtype), target=target)

    # # Inspect the computational graph
    # print("Computational DAG:")
    # print(task.compute_dag)

    log_file = f"bmm_bmm_{batch}-{M}-{N}-{K}-{L}-{in_dtype}-{acc_dtype}.json"
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
        a_np = np.random.uniform(size=(batch, M, K)).astype(in_dtype)
        b_np = np.random.uniform(size=(batch, K, L)).astype(in_dtype)
        c_np = np.random.uniform(size=(batch, L, N)).astype(in_dtype)
        out_np = np.random.uniform(size=(batch, M, N)).astype(in_dtype)
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
        a_np = np.random.uniform(size=(batch, M, K)).astype(in_dtype)
        b_np = np.random.uniform(size=(batch, K, L)).astype(in_dtype)
        c_np = np.random.uniform(size=(batch, L, N)).astype(in_dtype)
        out_np = np.random.uniform(size=(batch, M, N)).astype(in_dtype)
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
    python bmm_bmm_ansor.py --in_dtype float16 --acc_dtype float32 --begin 0 --num 1 --sm 80
"""

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),  # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)),  # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)),  # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512),  # Mixer-Large/32-S
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

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(
            batch=B,
            M=M,
            N=N,
            K=K,
            L=L,
            in_dtype=args.in_dtype,
            acc_dtype=args.acc_dtype,
            sm=args.sm,
            only_once=args.only_once,
            test=args.test
        )
        costs.append((ss, cost))

    print("B,M,N,K,L,in_dtype,acc_dtype,sm,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.in_dtype},{args.acc_dtype},{args.sm},{cc[1]}"
        )