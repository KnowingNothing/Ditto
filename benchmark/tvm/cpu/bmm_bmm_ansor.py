import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
import argparse
import pickle as pkl
import tvm.testing

EVALUTE_SCHEDULE_INPUTS = None
REPEAT = 2000
peakflops = {'sc': 704, 'sccc': 2150, 'scccc': 2995}

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def bmm_bmm(batch, M, N, K, L, dtype):
    A = te.placeholder((batch, M, K), name="A", dtype=dtype)
    B = te.placeholder((batch, K, L), name="B", dtype=dtype)
    C = te.placeholder((batch, L, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    D = te.compute(
        (batch, M, L),
        lambda b, m, l: te.sum(A[b, m, k].astype(dtype) * B[b, k, l].astype(dtype), axis=k),
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
    #     lambda b, m, l: (exp[b, m, l] / sumv[b, m]).astype(dtype)
    # )
    rrl = te.reduce_axis((0, L), name="rrl")
    E = te.compute(
        (batch, M, N),
        lambda b, m, n: te.sum(D[b, m, rrl].astype(dtype) * C[b, rrl, n].astype(dtype), axis=rrl),
        name="E"
    )
    
    out = te.compute(
        (batch, M, N),
        lambda b, m, n: E[b, m, n].astype(dtype),
        name="out"
    )

    return [A, B, C, out]

def main(batch, M, N, K, L, dtype, server):
    target = tvm.target.Target(f"llvm -mcpu=skylake-avx512")
    task = tvm.auto_scheduler.SearchTask(func=bmm_bmm, args=(batch, M, N, K, L, dtype), target=target)

    # # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = f"bmm_bmm_{batch}-{M}-{N}-{K}-{L}-{dtype}.json"

    n_line = 0
    if os.path.isfile(log_file):
        with open(log_file, 'rb') as f:
            n_line = len(f.readlines())
    n_line = min(n_line, 999)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000 - n_line,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    A, B, C, out = args

    # print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

        # cost = evaluate_schedule(sch, args, [A, B, C], [out], sm)
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(batch, M, K)).astype(dtype)
    b_np = np.random.uniform(size=(batch, K, L)).astype(dtype)
    c_np = np.random.uniform(size=(batch, L, N)).astype(dtype)
    out_np = np.random.uniform(size=(batch, M, N)).astype(dtype)
    out_groundTuruth = (a_np @ b_np) @ c_np
    
    dev = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)
    func(a_tvm, b_tvm, c_tvm, out_tvm)
    tvm.testing.assert_allclose(out_tvm.numpy(), out_groundTuruth, rtol = 1e-3, atol = 1)

    evaluator = func.time_evaluator(
            func.entry_name, dev, min_repeat_ms=0, repeat=REPEAT, number = 1, f_preproc="cache_flush_cpu_non_first_arg"
    )
    cost = evaluator(a_tvm, b_tvm, c_tvm, out_tvm).mean
    workload = batch * (M * K * L + M * N * L)
    ratioToPeak = workload / 1e9 / cost / peakflops[server]
    return (cost, ratioToPeak)


example_text = """
 example:
    python bmm_bmm_ansor.py --dtype float32 --begin 0 --num 1
"""

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512), # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),   # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256), # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256), # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),   # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)), # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)), # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512), # Mixer-Large/32-S
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
            dtype=args.dtype,
            server = args.server
        )
        costs.append((ss, cost))

    print("B,M,N,K,L,dtype,sm,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]}"
        )
    with open("bmm_bmm_ansor.pkl", 'wb') as f:
        pkl.dump(costs, f)