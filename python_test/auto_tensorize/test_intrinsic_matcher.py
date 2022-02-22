import pytest
import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math
import numpy as np

MI = 16
NI = 16
KI = 16
WARP_SIZE = 32
IN_VEC = 4
OUT_VEC = 4


def BatchGemmSoftmaxGemm(
    batch=12, M=512, N=64, K=64, L=512, in_dtype="float16", acc_dtype="float32"
):
    assert M % MI == 0
    assert N % NI == 0
    assert K % KI == 0
    assert L % NI == 0
    assert L % KI == 0

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


def test_build_match_info():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    first_packed = at.cuda_wmma(scope="shared")

    first_match_info_choices = at.intrinsic_match(D_frag, first_packed)

    # TODO: manually choose the specific choice for now
    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    choice = []
    for choice_ in first_match_info_choices:
        if set(choice_) == set([mi, li, rki]):
            choice = choice_

    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(scope="global")

    second_match_info_choices = at.intrinsic_match(E_frag, second_packed)

    # TODO: manually choose the specific choice for now
    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    choice = []
    for choice_ in second_match_info_choices:
        if set(choice_) == set([mi, ni, rli]):
            choice = choice_

    second_match_info = at.match_info(choice, second_packed)

    print(first_match_info_choices)
    print(second_match_info_choices)
    print(first_match_info)
    print(second_match_info)


def test_tensorize_cuda():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)

    first_packed = at.cuda_wmma(scope="shared")

    first_match_info_choices = at.intrinsic_match(D_frag, first_packed)

    # TODO: manually choose the specific choice for now
    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    choice = []
    for choice_ in first_match_info_choices:
        if set(choice_) == set([mi, li, rki]):
            choice = choice_
            
    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(scope="global")

    second_match_info_choices = at.intrinsic_match(E_frag, second_packed)

    # TODO: manually choose the specific choice for now
    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    choice = []
    for choice_ in second_match_info_choices:
        if set(choice_) == set([mi, ni, rli]):
            choice = choice_

    second_match_info = at.match_info(choice, second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )

    V100 = hw.query_hw_param("gpu.cuda.V100")
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

    sch = at.tensorize_cuda(layer, tensorize_state, V100, tensorize_param)
    print(tvm.lower(sch, layer.schedule_tensors, simple_mode=True))
    func = tvm.build(sch, layer.schedule_tensors, "cuda")
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float16")
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float16")
        for y in outs
    ]

    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    print(f"Our code uses {cost} ms")


def test_build_match_info_with_restrictions():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    first_packed = at.cuda_wmma(scope="shared")

    first_match_info_choices = at.intrinsic_match(D_frag, first_packed, ["InnerMost", "SameRange"])

    choice = first_match_info_choices[0]

    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(scope="global")

    second_match_info_choices = at.intrinsic_match(E_frag, second_packed, ["InnerMost", "SameRange"])

    choice = second_match_info_choices[0]

    second_match_info = at.match_info(choice, second_packed)

    print(first_match_info_choices)
    print(second_match_info_choices)
    print(first_match_info)
    print(second_match_info)


def test_tensorize_cuda_with_restrictions():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)

    first_packed = at.cuda_wmma(scope="shared")

    first_match_info_choices = at.intrinsic_match(D_frag, first_packed, ["InnerMost", "SameRange"])

    choice = first_match_info_choices[0]

    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(scope="global")

    second_match_info_choices = at.intrinsic_match(E_frag, second_packed, ["InnerMost", "SameRange"])

    choice = second_match_info_choices[0]

    second_match_info = at.match_info(choice, second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )

    V100 = hw.query_hw_param("gpu.cuda.V100")
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

    sch = at.tensorize_cuda(layer, tensorize_state, V100, tensorize_param)
    print(tvm.lower(sch, layer.schedule_tensors, simple_mode=True))
    func = tvm.build(sch, layer.schedule_tensors, "cuda")
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float16")
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float16")
        for y in outs
    ]

    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    print(f"Our code uses {cost} ms")


if __name__ == "__main__":
    test_build_match_info()
    test_tensorize_cuda()
    test_build_match_info_with_restrictions()
    test_tensorize_cuda_with_restrictions()
