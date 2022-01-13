import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule
import numpy as np


@T.prim_func
def gemm_kernel(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 16], dtype="int8")
    B = T.match_buffer(b, [8, 16], dtype="int8")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.allocate([4], "int8", scope="local")
    MultiB = T.allocate([4], "int8", scope="local")
    Accum = T.allocate([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 4]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 4]
    T.evaluate(
        T.ptx_mma(
            "m8n8k16",
            "row",
            "col",
            "int8",
            "int8",
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

    A_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    C_np = np.random.uniform(-1, 1, [8, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.asnumpy()

    from tvm import testing

    testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)
