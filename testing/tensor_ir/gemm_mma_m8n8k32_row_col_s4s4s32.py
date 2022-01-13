import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule
import numpy as np


@T.prim_func
def gemm_kernel(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 32], dtype="int4")
    B = T.match_buffer(b, [8, 32], dtype="int4")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.allocate([8], "int4", scope="local")
    MultiB = T.allocate([8], "int4", scope="local")
    Accum = T.allocate([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(8):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 8]
    for mma_multi_b_col in T.vectorized(8):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 8]
    T.evaluate(
        T.ptx_mma(
            "m8n8k32",
            "row",
            "col",
            "int4",
            "int4",
            "int32",
            MultiA,
            0,
            MultiB,
            0,
            Accum,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = T.load(
            "int32", Accum, mma_accum_c_id
        )


if __name__ == "__main__":
    sch = tvm.tir.Schedule(gemm_kernel)
    print(sch.mod.script())
    cuda_mod = tvm.build(sch.mod, target="cuda")
    print(cuda_mod.imported_modules[0].get_source())

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([8, 32], "int4", ctx)
    B_tvm = tvm.nd.empty([8, 32], "int4", ctx)
    C_tvm = tvm.nd.empty([8, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
