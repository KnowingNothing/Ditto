import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
import argparse
import pickle as pkl
import tvm.testing
import time

EVALUTE_SCHEDULE_INPUTS = None
REPEAT = 500
peakflops = {'sc': 704, 'sccc': 2150, 'scccc': 41.6}

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def bmm_bmm(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(A[m, k] * B[k, n], axis = k),
        name="C"
    )

    return [A, B, C]

def main(M, N, K, dtype, server, n_iter):
    target = tvm.target.Target(f"llvm -mcpu=skylake-avx512")
    task = tvm.auto_scheduler.SearchTask(func=bmm_bmm, args=(M, N, K, dtype), target=target)

    # # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = f"bmm_bmm_-{M}-{N}-{K}-{dtype}.json"

    n_line = 0
    if os.path.isfile(log_file):
        with open(log_file, 'rb') as f:
            n_line = len(f.readlines())
    n_line = min(n_line, n_iter)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_iter - n_line,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    # Run auto-tuning (search)
    t0 = time.time()
    task.tune(tune_option)
    t1 = time.time()
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    A, B, C = args

    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))

        # cost = evaluate_schedule(sch, args, [A, B, C], [out], sm)
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(M, K)).astype(dtype)
    b_np = np.random.uniform(size=(K, N)).astype(dtype)
    c_np = a_np @ b_np
    
    dev = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    func(a_tvm, b_tvm, c_tvm)
    tvm.testing.assert_allclose(c_tvm.numpy(), c_np, rtol = 1e-3, atol = 1)

    evaluator = func.time_evaluator(
            func.entry_name, dev, min_repeat_ms=0, repeat=REPEAT, number = 1, f_preproc="cache_flush_cpu_non_first_arg"
    )
    cost = evaluator(a_tvm, b_tvm, c_tvm).mean
    workload = M * N * K
    ratioToPeak = workload / 1e9 / cost / peakflops[server]
    return (cost, ratioToPeak, t1 - t0)


example_text = """
 example:
    python bmm_bmm_ansor.py --dtype float32 --begin 0 --num 1
"""

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (M, N, K)
    (6, 64, 64)
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
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--server", type=str, choices=peakflops.keys()
    )
    parser.add_argument(
        "--n_iter", type=int, default = 1000
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        M, N, K = ss
        cost = main(
            M=M,
            N=N,
            K=K,
            dtype=args.dtype,
            server = args.server,
            n_iter = args.n_iter
        )
        costs.append((ss, cost))

    print("M,N,K,dtype,sm,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{args.dtype},{cc[1]}"
        )
    with open("bmm_bmm_ansor.pkl", 'wb') as f:
        pkl.dump(costs, f)