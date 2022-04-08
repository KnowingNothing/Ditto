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


def Conv3x3Conv1x1(
    batch,
    in_channel,
    H,
    W,
    out_channel1,
    out_channel2,
    stride1=1,
    stride2=1,
    in_dtype="float16",
    acc_dtype="float32",
):
    assert stride2 == 1
    P1 = H // stride1
    Q1 = uround(W // stride1, MI)
    P2 = P1
    Q2 = Q1
    assert in_channel % KI == 0
    assert out_channel1 % NI == 0
    assert out_channel1 % KI == 0
    assert out_channel2 % NI == 0
    assert out_channel2 % KI == 0
    # assume after padding
    Img = tvm.te.placeholder([batch, in_channel//KI, P1 * stride1 + 2, Q1 + stride1 + 2, KI], name="Img", dtype=in_dtype)
    # assume after padding
    W1 = tvm.te.placeholder(
        [3, 3, in_channel//KI, out_channel1//NI, KI, NI], name="W1", dtype=in_dtype
    )
    W2 = tvm.te.placeholder([out_channel1 // KI,  out_channel2 // NI, KI, NI], name="W2", dtype=in_dtype)

    Img_shared = tvm.te.compute(
        [batch, 3, 3, in_channel//KI, P1, Q1//MI, MI, KI],
        lambda b, r, s, ko, p, mo, mi, ki:
            Img[
                b,
                ko,
                p * stride1 + r,
                (mo * MI + mi) % Q1 * stride2 + s,
                ki
            ],
        name="Img_shared",
    )

    W1_shared = tvm.te.compute(
        [3, 3, in_channel//KI, out_channel1//NI, KI, NI],
        lambda r, s, ko, lo, ki, li: W1[r, s, ko, lo, ki, li],
        name="W1_shared",
    )

    rko = tvm.te.reduce_axis([0, in_channel // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    rr = tvm.te.reduce_axis([0, 3], "rr")
    rs = tvm.te.reduce_axis([0, 3], "rs")
    Conv1_frag = tvm.te.compute(
        [batch, out_channel1//NI, P1, Q1//MI, MI, NI],
        lambda b, lo, p, mo, mi, li: tvm.te.sum(
            Img_shared[b, rr, rs, rko, p, mo, mi, rki].astype(acc_dtype)
            * W1_shared[rr, rs, rko, lo, rki, li].astype(acc_dtype),
            axis=[rr, rs, rko, rki],
        ),
        name="Conv1_frag",
    )

    cast = tvm.te.compute(
        [batch, out_channel1//NI, P1, Q1//MI, MI, NI],
        lambda b, lo, p, mo, mi, li: Conv1_frag[b, lo, p, mo, mi, li].astype(in_dtype),
        name="cast",
    )

    W2_shared = tvm.te.compute(
        [out_channel1 // KI,  out_channel2 // NI, KI, NI],
        lambda lo, no, li, ni: W2[lo, no, li, ni],
        name="W2_shared",
    )

    rlo = tvm.te.reduce_axis([0, out_channel2 // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    Conv2_frag = tvm.te.compute(
        [batch, out_channel2//NI, P2, Q2//MI, MI, NI],
        lambda b, no, p, mo, mi, ni: tvm.te.sum(
            cast[b, rlo, p, mo, mi, rli].astype(acc_dtype)
            * W2_shared[rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="Conv2_frag",
    )

    Out = tvm.te.compute(
        [batch, out_channel2//NI, P2, Q2, NI],
        lambda b, no, p, q, ni: Conv2_frag[
            b, no, p, q//NI, q%NI, ni
        ].astype(in_dtype),
        name="Out",
    )

    return [Img, W1, W2], [Out]


def main(
    batch,
    in_channel,
    H,
    W,
    out_channel1,
    out_channel2,
    stride1=1,
    stride2=1,
    in_dtype="float16",
    acc_dtype="float32",
    sm="70",
    only_once=False,
):
    ins, outs = Conv3x3Conv1x1(
        batch,
        in_channel,
        H,
        W,
        out_channel1,
        out_channel2,
        stride1=stride1,
        stride2=stride2,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
    )
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    cast, C_ext = E_frag.op.input_tensors
    D_frag = cast.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, no, p, mo, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, no, p, mo, rlo, mi, ni, rli], 4)

    first_packed = at.cuda_wmma(
        M=MI, N=NI, K=KI, in_dtype=in_dtype, out_dtype=acc_dtype, scope="shared"
    )

    first_match_info_choices = at.intrinsic_match(
        D_frag, first_packed, ["InnerMost", "SameRange"]
    )

    first_choice = first_match_info_choices[0]

    first_match_info = at.match_info(first_choice, first_packed)

    second_packed = at.cuda_wmma(
        M=MI, N=NI, K=KI, in_dtype=in_dtype, out_dtype=acc_dtype, scope="global"
    )

    second_match_info_choices = at.intrinsic_match(
        E_frag, second_packed, ["InnerMost", "SameRange"]
    )

    second_choice = second_match_info_choices[0]

    second_match_info = at.match_info(second_choice, second_packed)

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
    

    # 0-3
    # tensorize_param = at.cuda_tensorize_param(
    #     warp_size=32,
    #     ty_size=2,
    #     tz_size=4,
    #     input_vector_len=8,
    #     serial_y=2,
    #     serial_z=4,
    #     block_rx=2,
    #     warp_rx=4,
    #     block_ry=1,
    #     warp_ry=4,
    #     unroll_steps=512,
    # )
    tensorize_param = at.cuda_tensorize_param(
        warp_size=32,
        ty_size=2,
        tz_size=2,
        input_vector_len=8,
        serial_y=2,
        serial_z=2,
        block_rx=1,
        warp_rx=2,
        block_ry=1,
        warp_ry=2,
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
    python conv3x3_conv1x1_cuda.py --in_dtype float16 --acc_dtype float32 --begin 0 --num 1
"""


shapes = [
    (64, 112, 112, 192, 128, 2, 1),  # Yolo
    (32, 147, 147, 64, 80, 2, 1), # Inception-V3
    (64, 56, 56, 128, 64, 1, 1), # Darknet-19
    (128, 28, 28, 256, 128, 1, 1), # Darknet-19
    (16, 227, 227, 64, 16, 4, 1), # Squeezenet-V1.1
]


def cal_ai(batch, C, H, W, K1, K2, stride1, stride2):
    return (batch*(H//stride1)*(W//stride1)*K1/(batch*(H//stride1)*(W//stride1)+K1),
            batch*(H//stride1//stride2)*(W//stride1//stride2)*K2/(batch*(H//stride1//stride2)*(W//stride1//stride2)+K2))


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
            C, H, W, K1, K2, stride1, stride2 = ss
            ai1, ai2 = cal_ai(args.batch, C, H, W, K1, K2, stride1, stride2)
            print(ss, ai1, ai2)
    else:
        costs = []
        for ss in shapes[args.begin : args.begin + args.num]:
            C, H, W, K1, K2, stride1, stride2 = ss
            cost = main(
                args.batch,
                C,
                H,
                W,
                K1,
                K2,
                stride1=stride1,
                stride2=stride2,
                in_dtype=args.in_dtype,
                acc_dtype=args.acc_dtype,
                sm=args.sm,
                only_once=args.only_once,
            )
            costs.append((ss, cost))

        print("batch,C,H,W,K1,K2,stride1,stride2,in_dtype,acc_dtype,sm,cost")
        for cc in costs:
            print(
                f"{args.batch},"
                + ",".join(map(str, cc[0]))
                + f",{args.in_dtype},{args.acc_dtype},{args.sm},{cc[1]}"
            )
        for cc in costs:
            print(cc[1])
