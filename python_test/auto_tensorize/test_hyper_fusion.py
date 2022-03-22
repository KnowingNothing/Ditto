import pytest
import tvm
from tvm import te
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


class MicroKernel:
    def __init__(self, N=1, K=4, H=3, W=8, C=1, R=3, S=3) -> None:
        self.N = N
        self.K = K
        self.H = H
        self.W = W
        self.C = C
        self.R = R
        self.S = S

    def RoundUp(self, N, K, H, W, C, R, S):
        f = lambda a, b: (a + b - 1) // b * b
        return [
            f(N, self.N),
            f(K, self.K),
            f(H, self.H),
            f(W, self.W),
            f(C, self.C),
            f(R, self.R),
            f(S, self.S),
        ]

    def Verify(self, N, K, H, W, C, R, S):
        return (
            N % self.N == 0
            and K % self.K == 0
            and H % self.H == 0
            and W % self.W == 0
            and R % self.R == 0
            and S % self.S == 0
        )


def conv_relu_conv(
    N,
    C0,
    P0,
    Q0,
    C1,
    R1,
    S1,
    C2,
    R2,
    S2,
    padding1=0,
    padding2=0,
    stride1=1,
    stride2=1,
    in_dtype="float32",
    acc_dtype="float32",
    mk1=MicroKernel(),
    mk2=MicroKernel(),
):
    """
            N K  H  W  C R S
    Conv1   N C1 P0 Q0 C0 R1 S1
    Conv1   N C2 P1 Q1 C1 R2 S2
    """
    P1 = (P0 + 2 * padding1 - R1) // stride1 + 1
    Q1 = (Q0 + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    assert C1 % mk1.K == 0
    assert C2 % mk2.K == 0
    assert P0 % mk1.H == 0
    assert P1 % mk2.H == 0
    assert Q0 % mk1.W == 0
    assert Q1 % mk2.W == 0
    assert C0 % mk1.C == 0
    assert C1 % mk2.C == 0

    Img = tvm.te.placeholder([N, C0, P0, Q0], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder([C1, C0, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([C2, C1, R2, S2], dtype=in_dtype, name="Weight2")

    Pad1 = tvm.te.compute(
        [N, C0, P0 + 2 * padding1, Q0 + 2 * padding1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding1, p < P0 + padding1, q >= padding1, q < Q0 + padding1
            ),
            Img[n, c, p - padding1, q - padding1],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad1",
    )

    r1 = tvm.te.reduce_axis((0, R1), name="rr1")
    s1 = tvm.te.reduce_axis((0, S1), name="rs1")
    co1 = tvm.te.reduce_axis((0, C0 // mk1.C), name="rco1")
    ci1 = tvm.te.reduce_axis((0, mk1.C), name="rci1")
    Conv1 = tvm.te.compute(
        [N // mk1.N, mk1.N, C1 // mk1.K, mk1.K, P1 // mk1.H, mk1.H, Q1 // mk1.W, mk1.W],
        lambda no1, ni1, ko1, ki1, ho1, hi1, wo1, wi1: tvm.te.sum(
            Pad1[
                no1 * mk1.N + ni1,
                co1 * mk1.C + ci1,
                ho1 * mk1.H + hi1 + r1,
                wo1 * mk1.W + wi1 + s1,
            ]
            * Weight1[ko1 * mk1.K + ki1, co1 * mk1.C + ci1, r1, s1],
            axis=[co1, ci1, r1, s1],
        ),
        name="conv1",
    )
    Conv1_rfact = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: Conv1[
            n // mk1.N,
            n % mk1.N,
            c // mk1.K,
            c % mk1.K,
            p // mk1.H,
            p % mk1.H,
            q // mk1.W,
            q % mk1.W,
        ],
        name="conv1_unfactored",
    )
    Conv1_relu = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            Conv1_rfact[n, c, p, q] > 0,
            Conv1_rfact[n, c, p, q],
            tvm.tir.const(0, in_dtype),
        ),
        name="relu",
    )

    Pad2 = tvm.te.compute(
        [N, C1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding2, p < P0 + padding2, q >= padding2, q < Q0 + padding2
            ),
            Conv1_relu[n, c, p - padding2, q - padding2],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad2",
    )

    r2 = tvm.te.reduce_axis((0, R2), name="rr2")
    s2 = tvm.te.reduce_axis((0, S2), name="rs2")
    co2 = tvm.te.reduce_axis((0, C1 // mk2.C), name="rco2")
    ci2 = tvm.te.reduce_axis((0, mk2.C), name="rci2")
    Conv2_fact = tvm.te.compute(
        [N // mk2.N, mk2.N, C2 // mk2.K, mk2.K, P2 // mk2.H, mk2.H, Q2 // mk2.W, mk2.W],
        lambda no2, ni2, ko2, ki2, ho2, hi2, wo2, wi2: tvm.te.sum(
            Pad2[
                no2 * mk2.N + ni2,
                co2 * mk2.C + ci2,
                ho2 * mk2.H + hi2 + r2,
                wo2 * mk2.W + wi2 + s2,
            ]
            * Weight2[ko2 * mk2.K + ki2, co2 * mk2.C + ci2, r2, s2],
            axis=[co2, ci2, r2, s2],
        ),
        name="conv2_fact",
    )
    Conv2 = tvm.te.compute(
        [N, C2, P2, Q2],
        lambda n, c, p, q: Conv2_fact[
            n // mk2.N,
            n % mk2.N,
            c // mk2.C,
            c % mk2.C,
            p // mk2.H,
            p % mk2.H,
            q // mk2.W,
            q % mk2.W,
        ],
        name="conv2",
    )
    # return (N,C0,P0,Q0,C1,R1,S1,C2,R2,S2,P1,Q1,P2,Q2),(Img, Weight1, Weight2), (Pad1, Conv1, Conv1_rfact, Conv1_relu, Pad2, Conv2)
    return [Img, Weight1, Weight2], [Conv2], [Conv1, Conv2_fact]


def intrin_conv(mk):
    """
        In = tvm.te.placeholder([N, C, H + R - 1, W + S - 1], dtype=in_dtype, name='In' + surfix)
    Weight = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name='Weight' + surfix)
    r = tvm.te.reduce_axis((0, R), name='rr' + surfix)
    s = tvm.te.reduce_axis((0, S), name='rs' + surfix)
    c = tvm.te.reduce_axis((0, C), name='rc' + surfix)
    Out = tvm.te.compute([N, K, H, W], lambda n,k,h,w:
        tvm.te.sum(In[n,c,h+r,w+s] * Weight[k,c,r,s], axis=[c,r,s]),
        name = 'Ou' + surfix
    )"""
    N_, K_, H_, W_, C_, R_, S_ = mk.N, mk.K, mk.H, mk.W, mk.C, mk.R, mk.S
    assert N_ == 1
    assert K_ == 4
    assert H_ == 3
    assert W_ == 8
    assert R_ == 3
    assert S_ == 3
    in_dtype = "float32"
    In_ = te.placeholder([N_, C_, H_ + R_ - 1, W_ + S_ - 1], name="in", dtype=in_dtype)
    Weight_ = te.placeholder([K_, C_, R_, S_], name="weight", dtype=in_dtype)
    r_ = tvm.te.reduce_axis((0, R_), name="rr")
    s_ = tvm.te.reduce_axis((0, S_), name="rs")
    co = tvm.te.reduce_axis((0, 1), name="rco")
    c_ = tvm.te.reduce_axis((0, C_), name="rci")
    Out_ = te.compute(
        [1, N_, 1, K_, 1, H_, 1, W_],
        lambda no, n_, ko, k_, ho, h_, wo, w_: te.sum(
            In_[n_, c_, h_ + r_, w_ + s_] * Weight_[k_, c_, r_, s_],
            axis=[co, c_, r_, s_],
        ),
        name="out",
    )
    #  [21119070, 158790, 402, 1],  value [350, 50, 10, 1],
    varnames = [
        "chrws",
        "hrws",
        "ws",
        "crs",
        "rs",
        "s",
        "khw",
        "hw",
        "w",
        "_nkhw",
        "_khw",
        "_hw",
        "_w",
    ]
    vars = {varname: tvm.te.var(varname) for varname in varnames}
    In_b = tvm.tir.decl_buffer(
        In_.shape,
        In_.dtype,
        name="In",
        offset_factor=1,
        strides=[vars["chrws"], vars["hrws"], vars["ws"], 1],
    )  # N C H+R-1 W+S-1
    Weight_b = tvm.tir.decl_buffer(
        Weight_.shape,
        Weight_.dtype,
        name="Weight",
        offset_factor=1,
        strides=[vars["crs"], vars["rs"], vars["s"], 1],
    )  # K C R S
    Out_b = tvm.tir.decl_buffer(
        Out_.shape,
        Out_.dtype,
        name="Out",
        offset_factor=1,
        strides=[
            vars["_nkhw"],
            vars["khw"],
            vars["_khw"],
            vars["hw"],
            vars["_hw"],
            vars["w"],
            vars["_w"],
            1,
        ],
    )  # N K H W

    def intrin_func(ins, outs):
        In__, Weight__ = ins
        Out__ = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            # extern "C" int conv_update_avx2(float *In, float *Weight, float *Out, int HW_s_, int W_s_, int C_in_,
            # int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_){
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "conv_update_avx2",
                    In__.access_ptr("r"),
                    Weight__.access_ptr("r"),
                    Out__.access_ptr("w"),
                    vars["hw"],
                    vars["w"],
                    C_,
                    vars["hrws"],
                    vars["ws"],
                    vars["crs"],
                    vars["rs"],
                    vars["s"],
                )
                # extern "C" int conv_update(float *In, float *Weight, float *Out, int KHW, int HW, int W, int CHRWS, int HRWS, int WS, int CRS, int RS, int S,
                # int N_, int K_, int H_, int W_, int C_, int R_, int S_)
                # tvm.tir.call_extern(
                #     "int32",
                #     "conv_update",
                #     In__.access_ptr('r'),
                #     Weight__.access_ptr('r'),
                #     Out__.access_ptr('w'),
                #     vars['khw'], vars['hw'], vars['w'], vars['chrws'], vars['hrws'], vars['ws'], vars['crs'],vars['rs'],vars['s'],C_
                # )
            )
            return ib.get()

        def _reduce_reset():
            # long long conv_reset(float *Out, long long K, long long H, long long W,
            #            long long N_, long long K_, long long H_, long long W_)
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "conv_reset",
                    Out__.access_ptr("w"),
                    vars["khw"],
                    vars["hw"],
                    vars["w"],
                    N_,
                    K_,
                    H_,
                    W_,
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(
        Out_.op, intrin_func, binds={In_: In_b, Weight_: Weight_b, Out_: Out_b}
    )


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


def test_build_fusion_choice():
    ins, outs = BatchGemmSoftmaxGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op):
        loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

        compute = intrin_wmma_gemm("float16", "float32")

        store = intrin_wmma_store_matrix("shared", "float32")

        packed = at.packed_intrinsic(
            loads,
            compute,
            store,
            ["wmma.matrix_a", "wmma.matrix_b"],
            "wmma.accumulator",
            "wmma.accumulator",
        )
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, impl="")
        return match_info 

    first_match_info = get_match_info(op1)

    second_match_info = get_match_info(op2)
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    V100 = hw.query_hw_param("gpu.cuda.V100")

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=V100, dtype="float32", simple_mode=-1
    )

    print(fuse_choice)


def test_build_match_info():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

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

    print(first_match_info)
    print(second_match_info)


def test_build_tensorize_hyper_fusion_state():
    ins, outs = BatchGemmSoftmaxGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op):
        loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

        compute = intrin_wmma_gemm("float16", "float32")

        store = intrin_wmma_store_matrix("glonal", "float32")

        packed = at.packed_intrinsic(
            loads,
            compute,
            store,
            ["wmma.matrix_a", "wmma.matrix_b"],
            "wmma.accumulator",
            "wmma.accumulator",
        )
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, impl="")
        return match_info 

    first_match_info = get_match_info(op1)

    second_match_info = get_match_info(op2)
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    V100 = hw.query_hw_param("gpu.cuda.V100")

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=V100, dtype="float32", simple_mode=-1
    )
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )
    print(tensorize_state)
    print(tensorize_state.summary(verbose=True))

def test_tensorize_cuda():
    ins, outs = BatchGemmSoftmaxGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, storeScope):
        loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

        compute = intrin_wmma_gemm("float16", "float32")

        store = intrin_wmma_store_matrix(storeScope, "float32")

        packed = at.packed_intrinsic(
            loads,
            compute,
            store,
            ["wmma.matrix_a", "wmma.matrix_b"],
            "wmma.accumulator",
            "wmma.accumulator",
        )
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, impl="")
        return match_info 

    first_match_info = get_match_info(op1, "shared")

    second_match_info = get_match_info(op2, "global")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    V100 = hw.query_hw_param("gpu.cuda.V100")

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=V100, dtype="float32", simple_mode=-1
    )
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

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

def test_tensorize_cuda_run():
    ins, outs = BatchGemmSoftmaxGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, storeScope):
        loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

        compute = intrin_wmma_gemm("float16", "float32")

        store = intrin_wmma_store_matrix(storeScope, "float32")

        packed = at.packed_intrinsic(
            loads,
            compute,
            store,
            ["wmma.matrix_a", "wmma.matrix_b"],
            "wmma.accumulator",
            "wmma.accumulator",
        )
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, impl="")
        return match_info 

    first_match_info = get_match_info(op1, "shared")

    second_match_info = get_match_info(op2, "global")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    V100 = hw.query_hw_param("gpu.cuda.V100")

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=V100, dtype="float32", simple_mode=-1
    )
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

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
    test_build_fusion_choice()
    test_build_tensorize_hyper_fusion_state()
    test_tensorize_cuda()
    test_tensorize_cuda_run()
