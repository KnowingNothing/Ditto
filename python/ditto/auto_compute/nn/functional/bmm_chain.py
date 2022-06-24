import tvm
import math


def BatchGemmSoftmaxGemmMMA(
    query, key, value, MI=16, NI=16, KI=16, in_dtype="float16", acc_dtype="float32"
):
    """
    BatchGemmSoftmaxGemmMMA:
        a fused attention function that uses MMA
    ---
    args
    ---
    query: tvm.te.Tensor
    key: tvm.te.Tensor
    value: tvm.te.Tensor
    MI: int
    NI: int
    KI: int
    in_dtype: str
    acc_dtype: str

    Returns
    ---
    tvm.te.Tensor
    """
    A = query
    B = key
    C = value
    batch, M, K = A.shape
    _batch, _K, L = B.shape
    _batch, _L, N = C.shape
    batch, M, N, K, L = map(int, [batch, M, N, K, L])
    assert M % MI == 0, f"M={M}, MI={MI}"
    assert N % NI == 0, f"N={N}, NI={NI}"
    assert K % KI == 0, f"K={K}, KI={KI}"
    assert L % NI == 0, f"L={L}, NI={NI}"
    assert L % KI == 0, f"L={L}, KI={KI}"

    A_shared = tvm.te.compute(
        [batch, M // MI, K // KI, MI, KI],
        lambda b, mo, ko, mi, ki: A[b, mo * MI + mi, ko * KI + ki].astype(in_dtype),
        name="A_shared",
    )

    B_shared = tvm.te.compute(
        [batch, K // KI, L // NI, KI, NI],
        lambda b, ko, lo, ki, li: B[b, ko * KI + ki, lo * NI + li].astype(in_dtype),
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

    return F