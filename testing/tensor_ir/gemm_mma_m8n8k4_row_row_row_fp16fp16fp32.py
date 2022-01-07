import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule


@T.prim_func
def gemm_kernel(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [1024, 1024], dtype="float16")
    B = T.match_buffer(b, [1024, 1024], dtype="float16")
    C = T.match_buffer(c, [1024, 1024], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 8)
    T.launch_thread(bcol, 16)
    T.launch_thread(tx, 128)  # 2x2 warps, each warp 32 threads
    # warp_id = (tx // 32)
    # lane_id = (tx % 32)
    # wp_base_row = ((tx // 32) // 2)
    # wp_base_col = ((tx // 32 % 2))
    # mma_multi_a_row = ((tx % 32) % 4 + 4 * ((tx % 32) // 16))
    # mma_multi_b_row = ((tx % 32) % 4)
    MultiA = T.alloc_buffer([4], "float16", scope="local")
    MultiB = T.alloc_buffer([4], "float16", scope="local")
    Accum = T.alloc_buffer([8], "float32", scope="local")
    for i in range(8):
        Accum[i] = T.float32(0)
    for k1 in range(32):
        # each threadblock handles 32 (128x64x32) gemm
        for k2 in range(8):
            # k3 = 4, 1024 = 32x8x4
            for wp_row in range(4):  # stride is 2
                for wp_col in range(2):  # stride is 2
                    # each warp handles 4x2 (16x16x4) mma
                    # global_a_row_idx = (brow * 128 + ((((tx // 32) // 2) * 16) + (wp_row * 2) * 16) + ((tx % 32) % 4 + 4 * ((tx % 32) // 16)))
                    for mma_multi_a_col in T.vectorized(4):
                        # global_a_col_idx = (k1 * 32 + k2 * 4 + mma_multi_a_col)
                        MultiA[mma_multi_a_col] = A[
                            (
                                brow * 128
                                + ((((tx // 32) // 2) * 16) + (wp_row * 2) * 16)
                                + ((tx % 32) % 4 + 4 * ((tx % 32) // 16))
                            ),
                            (k1 * 32 + k2 * 4 + mma_multi_a_col),
                        ]
                    # global_b_row_idx = (k1 * 32 + k2 * 4 + ((tx % 32) % 4))
                    for mma_multi_b_col in T.vectorized(4):
                        # global_b_col_idx = (bcol * 64 + ((((tx // 32 % 2)) * 16) + (wp_col * 2) * 16) + mma_multi_a_col + (4 * ((tx % 32) // 16)))
                        MultiB[mma_multi_b_col] = B[
                            (k1 * 32 + k2 * 4 + ((tx % 32) % 4)),
                            (
                                bcol * 64
                                + ((((tx // 32 % 2)) * 16) + (wp_col * 2) * 16)
                                + mma_multi_b_col
                                + (4 * ((tx % 32) // 16))
                            ),
                        ]
                    T.evaluate(
                        T.call_extern(
                            "mma_m8n8k4",
                            MultiA.access_ptr("r"),
                            MultiB.access_ptr("r"),
                            Accum.access_ptr("rw"),
                            dtype="float32",
                        )
                    )
                    # global_c_row_idx_outer = (brow * 128 + ((((tx // 32) // 2) * 16) + (wp_row * 2) * 16))
                    # global_c_col_idx_outer = (bcol * 64 + ((((tx // 32 % 2)) * 16) + (wp_col * 2) * 16))
                    for mma_accum_c_id in range(8):
                        # global_c_row_idx_inner = (
                        #     ((tx % 32) % 2)
                        #     + (mma_accum_c_id // 2 % 2)
                        #     + 4 * ((tx % 32) // 16)
                        # )
                        # global_c_col_idx_inner = (
                        #     mma_accum_c_id // 4 % 2
                        #     + (tx % 32) // 2 % 2
                        #     + mma_accum_c_id % 2
                        # )
                        C[
                            (
                                brow * 128
                                + ((((tx // 32) // 2) * 16) + (wp_row * 2) * 16)
                            )
                            + (
                                ((tx % 32) % 2)
                                + (mma_accum_c_id // 2 % 2)
                                + 4 * ((tx % 32) // 16)
                            ),
                            (bcol * 64 + ((((tx // 32 % 2)) * 16) + (wp_col * 2) * 16))
                            + (
                                mma_accum_c_id // 4 % 2
                                + (tx % 32) // 2 % 2
                                + mma_accum_c_id % 2
                            ),
                        ] = Accum[mma_accum_c_id]


if __name__ == "__main__":
    sch = tvm.tir.Schedule(gemm_kernel)
    print(sch.mod.script())
