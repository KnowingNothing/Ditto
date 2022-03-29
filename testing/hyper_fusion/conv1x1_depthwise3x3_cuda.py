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
                print(error)
                print(".E", end="", flush=True)
                results = 1e10

        return results


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


def Conv1x1Depthwise3x3(
    batch,
    in_channel,
    H,
    W,
    out_channel,
    factor=1,
    stride1=1,
    stride2=1,
    in_dtype="float16",
    acc_dtype="float32",
):
    assert stride1 == 1
    P1 = uround(H // stride1, MI)
    Q1 = uround(W // stride1, MI)
    P2 = uround(P1 // stride2, MI)
    Q2 = uround(Q1 // stride2, MI)
    fusePQ1 = P1 * Q1
    fusePQ2 = P2 * Q2
    assert in_channel % KI == 0
    assert out_channel % NI == 0
    assert out_channel % KI == 0
    # assume after padding
    Img = tvm.te.placeholder([batch, P1, Q1, in_channel], name="Img", dtype=in_dtype)
    W1 = tvm.te.placeholder([in_channel//KI, out_channel//NI, KI, NI], name="W1", dtype=in_dtype)
    W2 = tvm.te.placeholder(
        [out_channel, uround(3 * 3, KI) // KI, uround(factor, NI) // NI, KI, NI], name="W2", dtype=in_dtype
    )

    Img_shared = tvm.te.compute(
        [batch, fusePQ1 // MI, in_channel // KI, MI, KI],
        lambda b, mo, ko, mi, ki: 
            Img[
                b,
                (mo * MI + mi) // Q1 * stride1,
                (mo * MI + mi) % Q1 * stride2,
                ko * KI + ki,
            ],
        name="Img_shared",
    )

    W1_shared = tvm.te.compute(
        [in_channel // KI, out_channel // NI, KI, NI],
        lambda ko, lo, ki, li: W1[ko, lo, ki, li],
        name="W1_shared",
    )

    rko = tvm.te.reduce_axis([0, in_channel // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    Conv1_frag = tvm.te.compute(
        [batch, fusePQ1 // MI, out_channel // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            Img_shared[b, mo, rko, mi, rki].astype(acc_dtype)
            * W1_shared[rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="Conv1_frag",
    )

    cast = tvm.te.compute(
        [batch, fusePQ1 // MI, out_channel // NI, MI, NI],
        lambda b, mo, lo, mi, li: Conv1_frag[b, mo, lo, mi, li].astype(in_dtype),
        name="cast",
    )
    
    fract = tvm.te.compute(
        [batch, P1, Q1, out_channel],
        lambda b, p, q, oc: cast[b, (p * Q1 + q) // MI, oc // NI, (p * Q1 + q) % MI, oc % NI]
    )

    pad = tvm.te.compute(
        [batch, out_channel, fusePQ2 // MI, uround(3 * 3, KI) // KI, MI, KI],
        lambda b, oc, mo, lo, mi, li: tvm.tir.if_then_else(
            tvm.tir.all(
                ((mo * MI + mi) // Q2 * stride2 + (lo * KI + li) // 3) < P1,
                ((mo * MI + mi) % Q2 * stride2 + (lo * KI + li) % 3) < Q1
            ),
            fract[
                b,
                ((mo * MI + mi) // Q2 * stride2 + (lo * KI + li) // 3),
                ((mo * MI + mi) % Q2 * stride2 + (lo * KI + li) % 3),
                oc
            ],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad",
    )

    W2_shared = tvm.te.compute(
        [out_channel, uround(3 * 3, KI) // KI, uround(factor, NI) // NI, KI, NI],
        lambda b, lo, no, li, ni: W2[b, lo, no, li, ni],
        name="W2_shared",
    )

    rlo = tvm.te.reduce_axis([0, uround(3 * 3, KI) // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    Dep_frag = tvm.te.compute(
        [batch, out_channel, fusePQ2 // MI, uround(factor, NI) // NI, MI, NI],
        lambda b, oc, mo, no, mi, ni: tvm.te.sum(
            pad[b, oc, mo, rlo, mi, rli].astype(acc_dtype)
            * W2_shared[oc, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="Dep_frag",
    )

    Out = tvm.te.compute(
        [batch, P2, Q2, out_channel * uround(factor, NI)],
        lambda b, p, q, oc: Dep_frag[
            b, oc // factor, (p * Q2 + q) // MI, oc % factor // NI, (p * Q2 + q) % MI, oc % factor % NI
        ].astype(in_dtype),
        name="Out",
    )

    return [Img, W1, W2], [Out]


def main(
    batch,
    in_channel,
    H,
    W,
    out_channel,
    factor=1,
    stride1=1,
    stride2=1,
    in_dtype="float16",
    acc_dtype="float32",
    sm="70",
    only_once=False,
):
    ins, outs = Conv1x1Depthwise3x3(
        batch,
        in_channel,
        H,
        W,
        out_channel,
        factor=factor,
        stride1=stride1,
        stride2=stride2,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
    )
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    pad, C_ext = E_frag.op.input_tensors
    fract = pad.op.input_tensors[0]
    cast = fract.op.input_tensors[0]
    D_frag = cast.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, oc, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, oc, m, n, rlo, mi, ni, rli], 4)

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
        ty_size=1,
        tz_size=1,
        input_vector_len=4,
        serial_y=1,
        serial_z=1,
        block_rx=1,
        warp_rx=1,
        block_ry=1,
        warp_ry=1,
        unroll_steps=64,
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
    python conv1x1_depthwise3x3_cuda.py --in_dtype float16 --acc_dtype float32 --begin 0 --num 1
"""


shapes = [
    # (C, H, W, K, stride1, stride2)
    # mobilenet-v1
    # (32, 112, 112, 64, 1, 2),
    (64, 56, 56, 128, 1, 1),
    (128, 56, 56, 128, 1, 2),
    (128, 28, 28, 256, 1, 1),
    (256, 28, 28, 256, 1, 2),
    (256, 14, 14, 512, 1, 1),
    (512, 14, 14, 512, 1, 1),
    (512, 7, 7, 1024, 1, 2),
]


def cal_ai(batch, C, H, W, K, stride1, stride2, factor):
    return (
        (
            batch
            * (H // stride1)
            * (W // stride1)
            * K
            / (batch * (H // stride1) * (W // stride1) + K)
        ),
        (
            batch
            * (H // stride1 // stride2)
            * (W // stride1 // stride2)
            * uround(factor, NI)
            / (
                batch * (H // stride1 // stride2) * (W // stride1 // stride2)
                + uround(factor, NI)
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument(
        "--sm",
        type=str,
        choices=["70", "80", "86"],
        default="70",
    )
    parser.add_argument("--ai", action="store_true")

    args = parser.parse_args()

    if not (args.sm, args.in_dtype, args.acc_dtype) in supported_dtypes:
        print(
            f"Data type not supported: sm_{args.sm}, in_dtype={args.in_dtype}, out_dtype={args.acc_dtype}."
        )

    if args.ai:
        for ss in shapes[args.begin : args.begin + args.num]:
            C, H, W, K, stride1, stride2 = ss
            ai1, ai2 = cal_ai(args.batch, C, H, W, K, stride1, stride2, args.factor)
            print(ss, ai1, ai2)
    else:
        costs = []
        for ss in shapes[args.begin : args.begin + args.num]:
            C, H, W, K, stride1, stride2 = ss
            cost = main(
                args.batch,
                C,
                H,
                W,
                K,
                factor=args.factor,
                stride1=stride1,
                stride2=stride2,
                in_dtype=args.in_dtype,
                acc_dtype=args.acc_dtype,
                sm=args.sm,
                only_once=args.only_once,
            )
            costs.append((ss, cost))

        print("batch,C,H,W,K,factor,in_dtype,acc_dtype,sm,cost")
        for cc in costs:
            print(
                f"{args.batch}"
                + ",".join(map(str, cc[0]))
                + f"{args.factor}"
                + f",{args.in_dtype},{args.acc_dtype},{args.sm},{cc[1]}"
            )
        for cc in costs:
            print(cc[1])
