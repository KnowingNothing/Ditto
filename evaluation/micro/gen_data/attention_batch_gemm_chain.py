import pytest
import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math
import numpy as np

from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

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


def intrin_wmma_load_matrix_a(in_dtype):
    A = tvm.te.placeholder((MI, KI), name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=int(KI * 2), offset_factor=256
    )
    C = tvm.te.compute((MI, KI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_a",
        data_alignment=int(KI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                MI,
                NI,
                KI,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                KI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="load_a"
    )


def intrin_wmma_load_matrix_b(in_dtype):
    A = tvm.te.placeholder((KI, NI), name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=int(NI * 2), offset_factor=256
    )
    C = tvm.te.compute((KI, NI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_b",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                MI,
                NI,
                KI,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                NI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="load_b"
    )


def intrin_wmma_gemm(in_dtype, acc_dtype):
    A = tvm.te.placeholder((MI, KI), name="A", dtype=in_dtype)
    B = tvm.te.placeholder((KI, NI), name="B", dtype=in_dtype)
    k = tvm.te.reduce_axis((0, KI), name="k")
    C = tvm.te.compute(
        (MI, NI),
        lambda ii, jj: tvm.te.sum(
            A[ii, k].astype(acc_dtype) * B[k, jj].astype(acc_dtype), axis=k
        ),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        data_alignment=int(KI * 2),
        offset_factor=256,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    MI,
                    NI,
                    KI,
                    BC.elem_offset // 256,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, B: BB, C: BC}, name="gemm"
    )


def intrin_wmma_store_matrix(scope, acc_dtype):
    A = tvm.te.placeholder((MI, NI), name="A", dtype=acc_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )
    C = tvm.te.compute((MI, NI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, data_alignment=int(NI * 2), offset_factor=256
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                MI,
                NI,
                KI,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                NI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="store"
    )


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


class Metric(object):
    def __init__(self, locality, parallelism, recomputation):
        self.locality = locality
        self.parallelism = parallelism
        self.recomputation = recomputation


def cal_F1(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (N // TN) * (L // TL) * TM * K
    ReadB = B * (M // TM) * (N // TN) * (L // TL) * K * TL
    ReadD = B * (M // TM) * (N // TN) * (L // TL) * TL * TN
    WriteE = B * (M // TM) * (N // TN) * (L // TL) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F2(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (N // TN) * TM * K
    ReadB = B * (M // TM) * (N // TN) * K * L
    ReadD = B * (M // TM) * (N // TN) * L * TN
    WriteE = B * (M // TM) * (N // TN) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F3(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (L // TL) * TM * K
    ReadB = B * (M // TM) * (L // TL) * K * TL
    ReadD = B * (M // TM) * (L // TL) * TL * N
    WriteE = B * (M // TM) * (L // TL) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F4(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (N // TN) * (L // TL) * M * K
    ReadB = B * (N // TN) * (L // TL) * K * TL
    ReadD = B * (N // TN) * (L // TL) * TL * TN
    WriteE = B * (N // TN) * (L // TL) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F5(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * TM * K
    ReadB = B * (M // TM) * K * L
    ReadD = B * (M // TM) * L * N
    WriteE = B * (M // TM) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F6(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (N // TN) * M * K
    ReadB = B * (N // TN) * K * L
    ReadD = B * (N // TN) * L * TN
    WriteE = B * (N // TN) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F7(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (L // TL) * M * K
    ReadB = B * (L // TL) * K * TL
    ReadD = B * (L // TL) * TL * N
    WriteE = B * (L // TL) * M * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * 1,
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def tensorize_cuda(batch, M, N, K, L, ins, outs, fuse_choice, f_cal, name):
    TMs = [2 ** i for i in range(int(math.log2(MI)), int(math.log2(M)) + 1)]
    TNs = [2 ** i for i in range(int(math.log2(NI)), int(math.log2(N)) + 1)]
    TKs = [2 ** i for i in range(int(math.log2(KI)), int(math.log2(K)) + 1)]
    TLs = [2 ** i for i in range(int(math.log2(KI)), int(math.log2(L)) + 1)]

    # TMs = [2*1*MI]
    # TNs = [2*2*NI]
    # TKs = [1*2*KI]
    # TLs = [2*2*KI]

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("shared", "float32")

    first_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    first_match_info = at.match_info([mi, li, rki], first_packed)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("global", "float32")

    second_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    second_match_info = at.match_info([mi, ni, rli], second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )

    V100 = hw.query_hw_param("gpu.cuda.V100")

    fout = open(f"dataset_attention_batch_gemm_chain_{name}.csv", "w")
    print("id,valid,Tm,Tn,Tk,Tl,locality,parallelism,recomputation,cost", flush=True)
    print(
        "id,valid,Tm,Tn,Tk,Tl,locality,parallelism,recomputation,cost",
        file=fout,
        flush=True,
    )
    count = 0
    for tm in TMs:
        for tn in TNs:
            for tk in TKs:
                for tl in TLs:
                    count += 1
                    metric = f_cal(
                        batch,
                        M,
                        N,
                        K,
                        L,
                        tm,
                        tn,
                        tk,
                        tl,
                        2,
                        4,
                        108 * 32,
                        108,
                        128 * 1024,
                    )

                    serial_y = 2
                    serial_z = 1
                    warp_rx = 4
                    warp_ry = 4
                    ty_size = (tm // MI + serial_z - 1) // serial_z
                    tz_size = (tn // NI + serial_y - 1) // serial_y
                    block_rx = (tk // KI + warp_rx - 1) // warp_rx
                    block_ry = (tl // KI + warp_ry - 1) // warp_ry
                    tensorize_param = at.cuda_tensorize_param(
                        warp_size=32,
                        ty_size=ty_size,
                        tz_size=tz_size,
                        input_vector_len=4,
                        serial_y=serial_y,
                        serial_z=serial_z,
                        block_rx=block_rx,
                        warp_rx=warp_rx,
                        block_ry=block_ry,
                        warp_ry=warp_ry,
                        unroll_steps=512,
                    )
                    try:
                        sch = at.tensorize_cuda(
                            layer, tensorize_state, V100, tensorize_param
                        )
                    except Exception as e:
                        print(e)
                        cost = 1e10
                        continue
                    cost = evaluate_schedule(sch, layer.schedule_tensors, ins, outs)
                    print(
                        f"{count},{cost<1e10},{tm},{tn},{tk},{tl},{metric.locality},{metric.parallelism},{metric.recomputation},{cost}",
                        flush=True,
                    )
                    print(
                        f"{count},{cost<1e10},{tm},{tn},{tk},{tl},{metric.locality},{metric.parallelism},{metric.recomputation},{cost}",
                        file=fout,
                        flush=True,
                    )
    fout.close()


if __name__ == "__main__":
    batch = 12
    M = 512
    N = 64
    K = 64
    L = 512
    
    ins, outs = BatchGemmSoftmaxGemm(
        batch=batch, M=M, N=N, K=K, L=L, in_dtype="float16", acc_dtype="float32"
    )
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    F1 = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)
    F2 = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 2)
    F3 = at.fusion_choice(D_frag.op, E_frag.op, [b, m, rlo, n, mi, rli, ni], 2)
    F4 = at.fusion_choice(D_frag.op, E_frag.op, [b, n, rlo, m, ni, rli, mi], 2)
    F5 = at.fusion_choice(D_frag.op, E_frag.op, [b, m, rlo, n, mi, rli, ni], 1)
    F6 = at.fusion_choice(D_frag.op, E_frag.op, [b, n, rlo, m, ni, rli, mi], 1)
    F7 = at.fusion_choice(D_frag.op, E_frag.op, [b, rlo, m, n, rli, mi, ni], 1)
    
    fuse_choices = [
        F1, F2, F3, F4, F5, F6, F7
    ]
    funcs = [cal_F1, cal_F2, cal_F3, cal_F4, cal_F5, cal_F6, cal_F7]
    names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    
    for i in range(0, 7):
        tensorize_cuda(batch, M, N, K, L, ins, outs, fuse_choices[i], funcs[i], names[i])
