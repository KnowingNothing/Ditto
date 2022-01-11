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
            mma_multi_b_col + (4 * ((tx % 32) // 8)),
        ]
    T.evaluate(
        T.call_extern(
            "ptx_mma",
            "m8n8k4",
            "row",
            "row",
            "fp16",
            "fp16",
            "fp32",
            MultiA,
            0,
            MultiB,
            0,
            Accum,
            0,
            False,
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
            (tx % 32) % 4 // 2 * 2
            + (tx % 32) % 16 // 8 * 4
            + mma_accum_c_id % 2
            + mma_accum_c_id // 4 * 8,
        ] = T.load("float32", Accum, mma_accum_c_id)


@T.prim_func
def gemm_kernel_1024(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [1024, 1024], dtype="float16")
    B = T.match_buffer(b, [1024, 1024], dtype="float16")
    C = T.match_buffer(c, [1024, 1024], dtype="float32")
    ####################################################
    # kernel configuration
    ####################################################
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    # we will use threadblock gemm 128x64x32
    # so the grid size is 8x16 (8=1024/128, 16=1024/16)
    T.launch_thread(brow, 8)
    T.launch_thread(bcol, 16)
    # we will use 4 warps
    # so the number of thread is 128
    T.launch_thread(tx, 128)
    # warp id is tx//32
    # lane id is tx%32
    # each warp handles 8=4x2=(128/16/2)x(64/16/2) MMA
    # so the warp organization for output (128x64) is as following:
    # ###################################################
    # |-------|-------|-------|-------|
    # |warp 0 | warp 2|warp 0 | warp 2|
    # |-------|-------|-------|-------|
    # |warp 1 | warp 3|warp 1 | warp 3|
    # |-------|-------|-------|-------|
    # |warp 0 | warp 2|warp 0 | warp 2|
    # |-------|-------|-------|-------|
    # |warp 1 | warp 3|warp 1 | warp 3|
    # |-------|-------|-------|-------|
    # |warp 0 | warp 2|warp 0 | warp 2|
    # |-------|-------|-------|-------|
    # |warp 1 | warp 3|warp 1 | warp 3|
    # |-------|-------|-------|-------|
    # |warp 0 | warp 2|warp 0 | warp 2|
    # |-------|-------|-------|-------|
    # |warp 1 | warp 3|warp 1 | warp 3|
    # |-------|-------|-------|-------|
    # each block is a 16x16x4 gemm

    # shared memory for A is 128x32
    SharedA = T.allocate([128 * 32], "float16", scope="shared")
    # shared memory for B is 32x64
    SharedB = T.allocate([32 * 64], "float16", scope="shared")
    # register for A, B, and accumulator
    MultiA = T.allocate([4 * 4], "float16", scope="local")
    MultiB = T.allocate([2 * 4], "float16", scope="local")
    Accum = T.allocate([4 * 2 * 8], "float32", scope="local")

    # initialize accumulator
    for wp_row in range(4):
        for wp_col in range(2):
            for i in range(4):
                Accum[wp_row * 16 + wp_col * 8 + i] = T.float32(0)

    for k1 in range(32):
        # load SharedA and SharedB
        # for SharedA, the thread arragement is <Thread 128 x Serial 4 x Vector 8>
        for i in range(4):
            for j in T.vectorized(8):
                SharedA[tx * 32 + i * 8 + j] = A[brow * 128 + tx, k1 * 32 + i * 8 + j]
        # for SharedB, the thread arrangement is <Thread 32 x Thread 4 x Serial 2 x Vector 8>
        for i in range(2):
            for j in T.vectorized(8):
                SharedB[tx // 4 * 64 + tx % 4 * 16 + i * 8 + j] = B[
                    k1 * 32 + tx // 4, bcol * 64 + tx % 4 * 16 + i * 8 + j
                ]

        for k2 in range(8):
            # load MultiA and MultiB
            for wp_row in range(4):
                for mma_multi_a_col in range(4):
                    # warp id is tx//32
                    # lane id is tx%32
                    MultiA[wp_row * 4 + mma_multi_a_col] = T.load(
                        "float16",
                        SharedA,
                        (
                            ((tx // 32) % 2 * 16 + wp_row * 2 * 16)
                            + (
                                ((tx % 32) % 4)
                                + (
                                    4
                                    * (
                                        (((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2))
                                        % 4
                                    )
                                )
                            )
                        )
                        * 32
                        + (k2 * 4)
                        + mma_multi_a_col,
                    )
            for wp_col in range(2):
                for mma_multi_b_col in range(4):
                    # warp id is tx//32
                    # lane id is tx%32
                    MultiB[wp_col * 4 + mma_multi_b_col] = T.load(
                        "float16",
                        SharedB,
                        ((k2 * 4) + ((tx % 32) % 4)) * 64
                        + ((tx // 32) // 2 * 16 + wp_col * 2 * 16)
                        + (mma_multi_b_col + (4 * ((tx % 32) // 8))),
                    )
            for wp_row in range(4):
                for wp_col in range(2):
                    T.evaluate(
                        T.call_extern(
                            "ptx_mma",
                            "m8n8k4",
                            "row",
                            "row",
                            "fp16",
                            "fp16",
                            "fp32",
                            MultiA,
                            wp_row * 4,
                            MultiB,
                            wp_col * 4,
                            Accum,
                            wp_row * 16 + wp_col * 8,
                            False,
                            dtype="float32",
                        )
                    )
    for wp_row in range(4):
        for wp_col in range(2):
            for mma_accum_c_id in range(8):
                C[
                    (brow * 128 + (tx // 32) % 2 * 16 + wp_row * 2 * 16)
                    + (
                        ((tx % 32) % 2)
                        + ((mma_accum_c_id // 2 % 2) * 2)
                        + 4 * ((tx % 32) // 16)
                        + ((tx % 32) % 16 // 4) % 2 * 8
                    ),
                    (bcol * 64 + (tx // 32) // 2 * 16 + wp_col * 2 * 16)
                    + (
                        (tx % 32) % 4 // 2 * 2
                        + (tx % 32) % 16 // 8 * 4
                        + mma_accum_c_id % 2
                        + mma_accum_c_id // 4 * 8
                    ),
                ] = T.load("float32", Accum, wp_row * 16 + wp_col * 8 + mma_accum_c_id)


if __name__ == "__main__":
    sch = tvm.tir.Schedule(gemm_kernel_1024)
    print(sch.mod.script())
    cuda_mod = tvm.build(sch.mod, target="cuda")
    print(cuda_mod.imported_modules[0].get_source())

    A_np = np.random.uniform(-1, 1, [1024, 1024]).astype("float16")
    B_np = np.random.uniform(-1, 1, [1024, 1024]).astype("float16")
    C_np = np.random.uniform(-1, 1, [1024, 1024]).astype("float32")
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

    testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)
