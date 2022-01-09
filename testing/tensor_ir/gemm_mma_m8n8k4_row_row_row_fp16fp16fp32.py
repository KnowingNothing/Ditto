import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule
import numpy as np


@T.prim_func
def gemm_kernel_16(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 4], dtype="float16")
    B = T.match_buffer(b, [4, 16], dtype="float16")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.allocate([4], "float16", scope="local")
    MultiB = T.allocate([4], "float16", scope="local")
    Accum = T.allocate([8], "float32", scope="local")
    for i in range(8):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[
            ((tx % 32) % 4) + (4 * ((((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2)) % 4)),
            mma_multi_a_col,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) % 4,
            mma_multi_b_col
            + (4 * ((tx % 32) // 8)),
        ]
    T.evaluate(
        T.call_extern(
            "mma_m8n8k4_row_row_fp16fp16fp32",
            MultiA, #.access_ptr("r"),
            MultiB, #.access_ptr("r"),
            Accum, #.access_ptr("rw"),
            dtype="float32",
        )
    )
    # this follows the ptx doc
    # for mma_accum_c_id in range(8):
    #     C[
    #         ((tx % 32) % 2)
    #         + ((mma_accum_c_id // 2 % 2) * 2)
    #         + 4 * ((tx % 32) // 16)
    #         + ((tx % 32) % 16 // 4) % 2 * 8,
    #         ((mma_accum_c_id // 4 % 2) * 4)
    #         + ((tx % 32) // 2 % 2 * 2)
    #         + mma_accum_c_id % 2
    #         + ((tx % 32) % 16 // 4) // 2 * 8,
    #     ] = T.load("float32", Accum, mma_accum_c_id)
    # this follows the results
    for mma_accum_c_id in range(8):
        C[
            ((tx % 32) % 2)
            + ((mma_accum_c_id // 2 % 2) * 2)
            + 4 * ((tx % 32) // 16)
            + ((tx % 32) % 16 // 4) % 2 * 8,
            (tx % 32) % 4 // 2 * 2 + (tx % 32) % 16 // 8 * 4 + mma_accum_c_id % 2
            + mma_accum_c_id // 4 * 8,
        ] = T.load("float32", Accum, mma_accum_c_id)
        
        
@T.prim_func
def gemm_kernel_16_vec(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 4], dtype="float16")
    B = T.match_buffer(b, [4, 16], dtype="float16")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.allocate([4], "float16", scope="local")
    MultiB = T.allocate([4], "float16", scope="local")
    Accum = T.allocate([8], "float32", scope="local")
    for i in range(8):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[
            ((tx % 32) % 4) + (4 * ((((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2)) % 4)),
            mma_multi_a_col,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) % 4,
            mma_multi_b_col
            + (4 * ((tx % 32) // 8)),
        ]
    T.evaluate(
        T.call_extern(
            "mma_m8n8k4_row_row_fp16fp16fp32",
            MultiA, #.access_ptr("r"),
            MultiB, #.access_ptr("r"),
            Accum, #.access_ptr("rw"),
            dtype="float32",
        )
    )
    for mma_accum_c_id in T.vectorized(4):
        C[
            ((tx % 32) % 2)
            + ((mma_accum_c_id // 2 % 2) * 2)
            + 4 * ((tx % 32) // 16)
            + ((tx % 32) % 16 // 4) % 2 * 8,
            (tx % 32) % 4 // 2 * 2 + (tx % 32) % 16 // 8 * 4 + mma_accum_c_id % 2
            + mma_accum_c_id // 4 * 8,
        ] = T.load("float32", Accum, mma_accum_c_id)
    for mma_accum_c_id in T.vectorized(4):
        C[
            ((tx % 32) % 2)
            + (((mma_accum_c_id + 4) // 2 % 2) * 2)
            + 4 * ((tx % 32) // 16)
            + ((tx % 32) % 16 // 4) % 2 * 8,
            (tx % 32) % 4 // 2 * 2 + (tx % 32) % 16 // 8 * 4 + (mma_accum_c_id + 4) % 2
            + (mma_accum_c_id + 4) // 4 * 8,
        ] = T.load("float32", Accum, mma_accum_c_id + 4)


if __name__ == "__main__":
    sch = tvm.tir.Schedule(gemm_kernel_16)
    print(sch.mod.script())
    cuda_mod = tvm.build(sch.mod, target="cuda")
    print(cuda_mod.imported_modules[0].get_source())
    
    A_np = np.random.uniform(-1, 1, [16, 4]).astype("float16")
    B_np = np.random.uniform(-1, 1, [4, 16]).astype("float16")
    C_np = np.random.uniform(-1, 1, [16, 16]).astype("float32")
    # for i in range(16):
    #     for j in range(4):
    #         A_np[i, j] = i * 4 + j
    # for i in range(16):
    #     for j in range(4):
    #         B_np[j, i] = j * 16 + i
    
    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    
    golden = np.matmul(A_np.astype("float32"), B_np.astype("float32"))
    
    # for i in range(16):
    #     for j in range(16):
    #         print(golden[i, j], ",", end="")
    #     print()
    
    # print()
    C_numpy = C_tvm.asnumpy()
    # for i in range(16):
    #     for j in range(16):
    #         print(C_numpy[i, j], ",", end="")
    #     print()
    from tvm import testing
    testing.assert_allclose(golden, C_numpy)
