import tvm
from tvm.script import tir as T
from tvm.ir.module import IRModule


@T.prim_func
def opt_conv_tensorcore_normalize(A: T.handle, W: T.handle, Conv: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # var definition
    bx = T.env_thread("blockIdx.x")
    by = T.env_thread("blockIdx.y")
    bz = T.env_thread("blockIdx.z")
    tx = T.env_thread("threadIdx.x")
    ty = T.env_thread("threadIdx.y")
    tz = T.env_thread("threadIdx.z")
    # buffer definition
    Apad_shared = T.buffer_decl(
        [16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Apad_shared_wmma_matrix_a = T.buffer_decl(
        [16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    BA = T.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256
    )
    BB = T.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256
    )
    BC = T.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
    Conv_wmma_accumulator = T.buffer_decl(
        [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    W_shared = T.buffer_decl(
        [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_shared_wmma_matrix_b = T.buffer_decl(
        [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    buffer = T.buffer_decl([16, 16], dtype="float16", scope="shared", align=32, offset_factor=256)
    buffer_1 = T.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256
    )
    buffer_2 = T.buffer_decl([16, 16], dtype="float16", scope="shared", align=32, offset_factor=256)
    buffer_3 = T.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256
    )
    buffer_4 = T.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
    buffer_5 = T.buffer_decl([16, 16], align=32, offset_factor=256)
    A_1 = T.match_buffer(
        A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_1 = T.match_buffer(
        W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Conv_1 = T.match_buffer(
        Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    # body
    T.realize(Conv_1[0:16, 0:14, 0:14, 0:32, 0:16, 0:16], "")
    T.launch_thread(bz, 196)
    T.launch_thread(bx, 2)
    T.launch_thread(by, 4)
    T.launch_thread(ty, 4)
    T.launch_thread(tz, 2)
    T.realize(
        Conv_wmma_accumulator[
            ((bx * 8) + (ty * 2)) : (((bx * 8) + (ty * 2)) + 2),
            T.floordiv(bz, 14) : (T.floordiv(bz, 14) + 1),
            T.floormod(bz, 14) : (T.floormod(bz, 14) + 1),
            ((by * 8) + (tz * 4)) : (((by * 8) + (tz * 4)) + 4),
            0:16,
            0:16,
        ],
        "wmma.accumulator",
    )
    for n_c_init in T.serial(0, 2):
        for o_c_init in T.serial(0, 4):
            T.attr(
                [BC, Conv_wmma_accumulator],
                "buffer_bind_scope",
                T.tvm_tuple(
                    (n_c_init + ((bx * 8) + (ty * 2))),
                    1,
                    T.floordiv(bz, 14),
                    1,
                    T.floormod(bz, 14),
                    1,
                    (o_c_init + ((by * 8) + (tz * 4))),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            T.evaluate(
                T.tvm_fill_fragment(
                    BC.data,
                    16,
                    16,
                    16,
                    T.floordiv(BC.elem_offset, 256),
                    T.float32(0),
                    dtype="handle",
                )
            )
    for ic_outer in T.serial(0, 8):
        for kh in T.serial(0, 3):
            T.realize(
                Apad_shared[
                    (bx * 8) : ((bx * 8) + 8),
                    (T.floordiv(bz, 14) + kh) : ((T.floordiv(bz, 14) + kh) + 1),
                    T.floormod(bz, 14) : (T.floormod(bz, 14) + 3),
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    0:16,
                    0:16,
                ],
                "shared",
            )
            for ax2 in T.serial(0, 3):
                for ax3 in T.serial(0, 2):
                    for ax4_ax5_fused_outer in T.serial(0, 8):
                        T.launch_thread(tx, 32)
                        Apad_shared[
                            ((tz + (ty * 2)) + (bx * 8)),
                            (T.floordiv(bz, 14) + kh),
                            (ax2 + T.floormod(bz, 14)),
                            (ax3 + (ic_outer * 2)),
                            T.floordiv((tx + (ax4_ax5_fused_outer * 32)), 16),
                            T.floormod((tx + (ax4_ax5_fused_outer * 32)), 16),
                        ] = T.if_then_else(
                            (
                                (
                                    (
                                        ((T.floordiv(bz, 14) + kh) >= 1)
                                        and (((T.floordiv(bz, 14) + kh) - 1) < 14)
                                    )
                                    and ((ax2 + T.floormod(bz, 14)) >= 1)
                                )
                                and (((ax2 + T.floormod(bz, 14)) - 1) < 14)
                            ),
                            A_1[
                                ((tz + (ty * 2)) + (bx * 8)),
                                ((T.floordiv(bz, 14) + kh) - 1),
                                ((ax2 + T.floormod(bz, 14)) - 1),
                                (ax3 + (ic_outer * 2)),
                                T.floordiv((tx + (ax4_ax5_fused_outer * 32)), 16),
                                T.floormod((tx + (ax4_ax5_fused_outer * 32)), 16),
                            ],
                            T.float16(0),
                            dtype="float16",
                        )
            T.realize(
                W_shared[
                    kh : (kh + 1),
                    0:3,
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    (by * 8) : ((by * 8) + 8),
                    0:16,
                    0:16,
                ],
                "shared",
            )
            for ax1 in T.serial(0, 3):
                for ax2_1 in T.serial(0, 2):
                    T.launch_thread(tx, 32)
                    for ax4_ax5_fused_inner in T.vectorized(0, 8):
                        W_shared[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((tz + (ty * 2)) + (by * 8)),
                            T.floordiv((ax4_ax5_fused_inner + (tx * 8)), 16),
                            T.floormod((ax4_ax5_fused_inner + (tx * 8)), 16),
                        ] = W_1[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((tz + (ty * 2)) + (by * 8)),
                            T.floordiv((ax4_ax5_fused_inner + (tx * 8)), 16),
                            T.floormod((ax4_ax5_fused_inner + (tx * 8)), 16),
                        ]
            for ic_inner in T.serial(0, 2):
                for kw in T.serial(0, 3):
                    T.realize(
                        Apad_shared_wmma_matrix_a[
                            ((bx * 8) + (ty * 2)) : (((bx * 8) + (ty * 2)) + 2),
                            (T.floordiv(bz, 14) + kh) : ((T.floordiv(bz, 14) + kh) + 1),
                            (kw + T.floormod(bz, 14)) : ((kw + T.floormod(bz, 14)) + 1),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            0:16,
                            0:16,
                        ],
                        "wmma.matrix_a",
                    )
                    for ax0 in T.serial(0, 2):
                        T.attr(
                            [buffer, Apad_shared],
                            "buffer_bind_scope",
                            T.tvm_tuple(
                                (ax0 + ((bx * 8) + (ty * 2))),
                                1,
                                (T.floordiv(bz, 14) + kh),
                                1,
                                (kw + T.floormod(bz, 14)),
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        T.attr(
                            [buffer_1, Apad_shared_wmma_matrix_a],
                            "buffer_bind_scope",
                            T.tvm_tuple(
                                (ax0 + ((bx * 8) + (ty * 2))),
                                1,
                                (T.floordiv(bz, 14) + kh),
                                1,
                                (kw + T.floormod(bz, 14)),
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                buffer_1.data,
                                16,
                                16,
                                16,
                                T.floordiv(buffer_1.elem_offset, 256),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    buffer.data,
                                    buffer.elem_offset,
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                    T.realize(
                        W_shared_wmma_matrix_b[
                            kh : (kh + 1),
                            kw : (kw + 1),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            ((by * 8) + (tz * 4)) : (((by * 8) + (tz * 4)) + 4),
                            0:16,
                            0:16,
                        ],
                        "wmma.matrix_b",
                    )
                    for ax3_1 in T.serial(0, 4):
                        T.attr(
                            [buffer_2, W_shared],
                            "buffer_bind_scope",
                            T.tvm_tuple(
                                kh,
                                1,
                                kw,
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                (ax3_1 + ((by * 8) + (tz * 4))),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        T.attr(
                            [buffer_3, W_shared_wmma_matrix_b],
                            "buffer_bind_scope",
                            T.tvm_tuple(
                                kh,
                                1,
                                kw,
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                (ax3_1 + ((by * 8) + (tz * 4))),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                buffer_3.data,
                                16,
                                16,
                                16,
                                T.floordiv(buffer_3.elem_offset, 256),
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    buffer_2.data,
                                    buffer_2.elem_offset,
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                    for n_c in T.serial(0, 2):
                        for o_c in T.serial(0, 4):
                            T.attr(
                                [BA, Apad_shared_wmma_matrix_a],
                                "buffer_bind_scope",
                                T.tvm_tuple(
                                    (n_c + ((bx * 8) + (ty * 2))),
                                    1,
                                    (T.floordiv(bz, 14) + kh),
                                    1,
                                    (T.floormod(bz, 14) + kw),
                                    1,
                                    ((ic_outer * 2) + ic_inner),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            T.attr(
                                [BB, W_shared_wmma_matrix_b],
                                "buffer_bind_scope",
                                T.tvm_tuple(
                                    kh,
                                    1,
                                    kw,
                                    1,
                                    ((ic_outer * 2) + ic_inner),
                                    1,
                                    (o_c + ((by * 8) + (tz * 4))),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            T.attr(
                                [BC, Conv_wmma_accumulator],
                                "buffer_bind_scope",
                                T.tvm_tuple(
                                    (n_c + ((bx * 8) + (ty * 2))),
                                    1,
                                    T.floordiv(bz, 14),
                                    1,
                                    T.floormod(bz, 14),
                                    1,
                                    (o_c + ((by * 8) + (tz * 4))),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            T.evaluate(
                                T.tvm_mma_sync(
                                    BC.data,
                                    T.floordiv(BC.elem_offset, 256),
                                    BA.data,
                                    T.floordiv(BA.elem_offset, 256),
                                    BB.data,
                                    T.floordiv(BB.elem_offset, 256),
                                    BC.data,
                                    T.floordiv(BC.elem_offset, 256),
                                    dtype="handle",
                                )
                            )
    for n_inner in T.serial(0, 2):
        for o_inner in T.serial(0, 4):
            T.attr(
                [buffer_4, Conv_wmma_accumulator],
                "buffer_bind_scope",
                T.tvm_tuple(
                    ((((bx * 4) + ty) * 2) + n_inner),
                    1,
                    T.floordiv(bz, 14),
                    1,
                    T.floormod(bz, 14),
                    1,
                    ((((by * 2) + tz) * 4) + o_inner),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            T.attr(
                [buffer_5, Conv_1],
                "buffer_bind_scope",
                T.tvm_tuple(
                    ((((bx * 4) + ty) * 2) + n_inner),
                    1,
                    T.floordiv(bz, 14),
                    1,
                    T.floormod(bz, 14),
                    1,
                    ((((by * 2) + tz) * 4) + o_inner),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            T.evaluate(
                T.tvm_store_matrix_sync(
                    buffer_4.data,
                    16,
                    16,
                    16,
                    T.floordiv(buffer_4.elem_offset, 256),
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float32"),
                        buffer_5.data,
                        buffer_5.elem_offset,
                        256,
                        2,
                        dtype="handle",
                    ),
                    16,
                    "row_major",
                    dtype="handle",
                )
            )


def test_opt_conv_tensorcore_normalize():
    mod = opt_conv_tensorcore_normalize
    print(type(mod))
    ir_mod = IRModule({"main": mod})
    print(type(ir_mod))
    mod = tvm.build(ir_mod, "cuda")
    

if __name__ == "__main__":
    test_opt_conv_tensorcore_normalize()