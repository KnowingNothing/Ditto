import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule
import numpy as np


@T.prim_func
def gemm_kernel(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 4], dtype="float64")
    B = T.match_buffer(b, [8, 4], dtype="float64")
    C = T.match_buffer(c, [8, 8], dtype="float64")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.allocate([1], "float64", scope="local")
    MultiB = T.allocate([1], "float64", scope="local")
    Accum = T.allocate([2], "float64", scope="local")
    for i in range(2):
        Accum[i] = T.float64(0)

    MultiA[0] = A[(tx % 32) // 4, (tx % 32) % 4]
    MultiB[0] = B[(tx % 32) // 4, (tx % 32) % 4]
    T.evaluate(
        T.ptx_mma(
            "m8n8k4",
            "row",
            "col",
            "fp64",
            "fp64",
            "fp64",
            MultiA,
            0,
            MultiB,
            0,
            Accum,
            0,
            False,
            dtype="float64",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = T.load(
            "float64", Accum, mma_accum_c_id
        )


if __name__ == "__main__":
    sch = tvm.tir.Schedule(gemm_kernel)
    print(sch.mod.script())
    cuda_mod = tvm.build(sch.mod, target="cuda")
    print(cuda_mod.imported_modules[0].get_source())

    A_np = np.random.uniform(-1, 1, [8, 4]).astype("float64")
    B_np = np.random.uniform(-1, 1, [8, 4]).astype("float64")
    C_np = np.random.uniform(-1, 1, [8, 8]).astype("float64")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float64"), B_np.astype("float64").T)

    C_numpy = C_tvm.asnumpy()
    from tvm import testing

    testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)
