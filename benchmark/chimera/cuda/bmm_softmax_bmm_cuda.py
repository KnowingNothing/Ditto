import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math
import numpy as np
import argparse

from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

MI = 16
NI = 16
KI = 16
WARP_SIZE = 32
IN_VEC = 4
OUT_VEC = 4

EVALUTE_SCHEDULE_INPUTS = None


def evaluate_schedule_worker(dummy):
    global EVALUTE_SCHEDULE_INPUTS
    sch, args, ins, outs = EVALUTE_SCHEDULE_INPUTS
    func = tvm.build(sch, args, "cuda")
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


def evaluate_schedule(sch, args, ins, outs):
    global EVALUTE_SCHEDULE_INPUTS
    EVALUTE_SCHEDULE_INPUTS = (sch, args, ins, outs)
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
                # print(".T", end="", flush=True)
                results = 1e10
            except Exception as error:
                # print(error)
                # print(".E", end="", flush=True)
                results = 1e10

        return results


def BatchGemmSoftmaxGemm(
    batch=12, M=512, N=64, K=64, L=512, in_dtype="float16", acc_dtype="float32"
):
    assert M % MI == 0, f"M={M}"
    assert N % NI == 0, f"N={N}"
    assert K % KI == 0, f"K={K}"
    assert L % NI == 0, f"L={L}"
    assert L % KI == 0, f"L={L}"
    # def upper_round(x, b):
    #     return (x + b - 1) // b * b

    # M = upper_round(M, MI)
    # N = upper_round(N, NI)
    # K = upper_round(K, KI)
    # L = upper_round(L, NI)
    # L = upper_round(L, KI)

    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=in_dtype)

    A_shared = tvm.te.compute(
        [batch, M // MI, K // KI, MI, KI],
        lambda b, mo, ko, mi, ki: A[b, mo * MI + mi, ko * KI + ki],
        name="A_shared",
    )

    B_shared = tvm.te.compute(
        [batch, K // KI, L // NI, KI, NI],
        lambda b, ko, lo, ki, li: B[b, ko * KI + ki, lo * NI + li],
        name="B_shared",
    )

    rko = tvm.te.reduce_axis([0, K // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A_shared[b, mo, rko, mi, rki].astype(acc_dtype)
            * B_shared[b, rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    exp = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.exp(D_frag[b, mo, lo, mi, li]).astype(
            in_dtype
        ),
        name="exp",
    )

    ext_N = 2 ** math.ceil(math.log2(N // NI + 1)) * NI
    C_ext = tvm.te.compute(
        [batch, L // KI, ext_N // NI, KI, NI],
        lambda b, lo, no, li, ni: tvm.tir.if_then_else(
            no * NI + ni < N,
            C[b, lo * NI + li, no * NI + ni],
            tvm.tir.const(1, in_dtype),
        ),
        name="C_ext",
    )

    rlo = tvm.te.reduce_axis([0, L // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, ext_N // NI, MI, NI],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            exp[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_ext[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, n // NI, m % MI, n % NI].astype(in_dtype)
        / (
            E_frag[b, m // MI, ext_N // NI - 1, m % MI, NI - 1].astype(in_dtype)
            + tvm.tir.const(1e-5, in_dtype)
        ),
        name="F",
    )

    return [A, B, C], [F]


def main(
    batch=12,
    M=512,
    N=64,
    K=64,
    L=512,
    in_dtype="float16",
    acc_dtype="float32",
    sm="70",
    only_once=False,
):
    ins, outs = BatchGemmSoftmaxGemm(
        batch=batch, M=M, N=N, K=K, L=L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)

    first_packed = at.cuda_wmma(
        M=MI, N=NI, K=KI, in_dtype=in_dtype, out_dtype=acc_dtype, scope="shared"
    )

    first_match_info_choices = at.intrinsic_match(
        D_frag, first_packed, ["InnerMost", "SameRange"]
    )

    choice = first_match_info_choices[0]

    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(
        M=MI, N=NI, K=KI, in_dtype=in_dtype, out_dtype=acc_dtype, scope="global"
    )

    second_match_info_choices = at.intrinsic_match(
        E_frag, second_packed, ["InnerMost", "SameRange"]
    )

    choice = second_match_info_choices[0]

    second_match_info = at.match_info(choice, second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )

    if sm == "70":
        device = hw.query_hw_param("gpu.cuda.V100")
    elif sm == "80":
        device = hw.query_hw_param("gpu.cuda.A100")
    elif sm == "86":
        device = hw.query_hw_param("gpu.cuda.RTX3090")
    else:
        raise RuntimeError(f"SM_{sm} is not supported.")

    tensorize_param = at.cuda_tensorize_param(
        warp_size=32,
        ty_size=4,
        tz_size=2,
        input_vector_len=4,
        serial_y=2,
        serial_z=1,
        block_rx=8,
        warp_rx=4,
        block_ry=1,
        warp_ry=4,
        unroll_steps=512,
    )

    sch = at.tensorize_cuda(layer, tensorize_state, device, tensorize_param)
    if only_once:
        print(tvm.lower(sch, layer.schedule_tensors, simple_mode=True))
        func = tvm.build(sch, layer.schedule_tensors, "cuda")
        inputs_np = [
            np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
            for y in ins
        ]

        outputs_np = [
            np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
            for y in outs
        ]

        ctx = tvm.cuda()
        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        # evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
        # cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        func(*inputs_tvm, *outputs_tvm)
        cost = -1
    else:
        cost = evaluate_schedule(sch, layer.schedule_tensors, ins, outs)
    # print(f"Our code uses {cost} ms")
    return cost


supported_dtypes = set(
    [  # sm, in_dtype, acc_dtype
        ("70", "float16", "float16"),
        ("70", "float16", "float32"),
        ("80", "float16", "float16"),
        ("80", "float16", "float32"),
        ("80", "float64", "float64"),
        ("80", "int4", "int32"),
        ("80", "int8", "int32"),
        ("86", "float16", "float16"),
        ("86", "float16", "float32"),
        ("86", "float64", "float64"),
        ("86", "int4", "int32"),
        ("86", "int8", "int32"),
    ]
)

example_text = """
 example:
    python bmm_softmax_bmm_cuda.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1
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
    parser.add_argument("--profile", action="store_true")
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

    if not (args.sm, args.in_dtype, args.acc_dtype) in supported_dtypes:
        print(
            f"Data type not supported: sm_{args.sm}, in_dtype={args.in_dtype}, out_dtype={args.acc_dtype}."
        )

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
        )
        costs.append((ss, cost))

    print("B,M,N,K,in_dtype,acc_dtype,sm,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{args.in_dtype},{args.acc_dtype},{args.sm},{cc[1]}"
        )
