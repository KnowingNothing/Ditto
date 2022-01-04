import tvm
from .cuda_dtype import cuda_bf16, cuda_tf32
from ... import visa


# TODO: measure the latency of these isa
# binary add
cuda_add_fp32_fp32_fp32 = visa.scalar_binary_add(
    1.0, "float32", "float32", "float32", name="cuda.add.fp32.fp32.fp32")
cuda_add_fp16_fp16_fp32 = visa.scalar_binary_add(
    1.0, "float16", "float16", "float32", name="cuda.add.fp16.fp16.fp32")
cuda_add_fp16_fp16_fp16 = visa.scalar_binary_add(
    1.0, "float16", "float16", "float16", name="cuda.add.fp16.fp16.fp16")
cuda_add_int32_int32_int32 = visa.scalar_binary_add(
    1.0, "int32", "int32", "int32", name="cuda.add.int32.int32.int32")
cuda_add_int8_int8_int32 = visa.scalar_binary_add(
    1.0, "int8", "int8", "int32", name="cuda.add.int8.int8.int32")
cuda_add_int8_int8_int8 = visa.scalar_binary_add(
    1.0, "int8", "int8", "int8", name="cuda.add.int8.int8.int8")

# binary sub
cuda_sub_fp32_fp32_fp32 = visa.scalar_binary_sub(
    1.0, "float32", "float32", "float32", name="cuda.sub.fp32.fp32.fp32")
cuda_sub_fp16_fp16_fp32 = visa.scalar_binary_sub(
    1.0, "float16", "float16", "float32", name="cuda.sub.fp16.fp16.fp32")
cuda_sub_fp16_fp16_fp16 = visa.scalar_binary_sub(
    1.0, "float16", "float16", "float16", name="cuda.sub.fp16.fp16.fp16")
cuda_sub_int32_int32_int32 = visa.scalar_binary_sub(
    1.0, "int32", "int32", "int32", name="cuda.sub.int32.int32.int32")
cuda_sub_int8_int8_int32 = visa.scalar_binary_sub(
    1.0, "int8", "int8", "int32", name="cuda.sub.int8.int8.int32")
cuda_sub_int8_int8_int8 = visa.scalar_binary_sub(
    1.0, "int8", "int8", "int8", name="cuda.sub.int8.int8.int8")

# binary mul
cuda_mul_fp32_fp32_fp32 = visa.scalar_binary_mul(
    2.0, "float32", "float32", "float32", name="cuda.mul.fp32.fp32.fp32")
cuda_mul_fp16_fp16_fp32 = visa.scalar_binary_mul(
    2.0, "float16", "float16", "float32", name="cuda.mul.fp16.fp16.fp32")
cuda_mul_fp16_fp16_fp16 = visa.scalar_binary_mul(
    2.0, "float16", "float16", "float16", name="cuda.mul.fp16.fp16.fp16")
cuda_mul_int32_int32_int32 = visa.scalar_binary_mul(
    2.0, "int32", "int32", "int32", name="cuda.mul.int32.int32.int32")
cuda_mul_int8_int8_int32 = visa.scalar_binary_mul(
    2.0, "int8", "int8", "int32", name="cuda.mul.int8.int8.int32")
cuda_mul_int8_int8_int8 = visa.scalar_binary_mul(
    2.0, "int8", "int8", "int8", name="cuda.mul.int8.int8.int8")

# binary div
cuda_div_fp32_fp32_fp32 = visa.scalar_binary_div(
    2.0, "float32", "float32", "float32", name="cuda.div.fp32.fp32.fp32")
cuda_div_fp16_fp16_fp32 = visa.scalar_binary_div(
    2.0, "float16", "float16", "float32", name="cuda.div.fp16.fp16.fp32")
cuda_div_fp16_fp16_fp16 = visa.scalar_binary_div(
    2.0, "float16", "float16", "float16", name="cuda.div.fp16.fp16.fp16")
cuda_div_int32_int32_int32 = visa.scalar_binary_div(
    2.0, "int32", "int32", "int32", name="cuda.div.int32.int32.int32")
cuda_div_int8_int8_int32 = visa.scalar_binary_div(
    2.0, "int8", "int8", "int32", name="cuda.div.int8.int8.int32")
cuda_div_int8_int8_int8 = visa.scalar_binary_div(
    2.0, "int8", "int8", "int8", name="cuda.div.int8.int8.int8")

# binary mod
cuda_mod_fp32_fp32_fp32 = visa.scalar_binary_mod(
    2.0, "float32", "float32", "float32", name="cuda.mod.fp32.fp32.fp32")
cuda_mod_fp16_fp16_fp32 = visa.scalar_binary_mod(
    2.0, "float16", "float16", "float32", name="cuda.mod.fp16.fp16.fp32")
cuda_mod_fp16_fp16_fp16 = visa.scalar_binary_mod(
    2.0, "float16", "float16", "float16", name="cuda.mod.fp16.fp16.fp16")
cuda_mod_int32_int32_int32 = visa.scalar_binary_mod(
    2.0, "int32", "int32", "int32", name="cuda.mod.int32.int32.int32")
cuda_mod_int8_int8_int32 = visa.scalar_binary_mod(
    2.0, "int8", "int8", "int32", name="cuda.mod.int8.int8.int32")
cuda_mod_int8_int8_int8 = visa.scalar_binary_mod(
    2.0, "int8", "int8", "int8", name="cuda.mod.int8.int8.int8")

# multiply-add
cuda_fma_fp32_fp32_fp32 = visa.scalar_multiply_add(
    2.5, "float32", "float32", "float32", name="cuda.fma.fp32.fp32.fp32")
cuda_fma_fp16_fp16_fp32 = visa.scalar_multiply_add(
    2.5, "float16", "float16", "float32", name="cuda.fma.fp16.fp16.fp32")
cuda_fma_fp16_fp16_fp16 = visa.scalar_multiply_add(
    2.5, "float16", "float16", "float16", name="cuda.fma.fp16.fp16.fp16")
cuda_fma_int32_int32_int32 = visa.scalar_multiply_add(
    2.5, "int32", "int32", "int32", name="cuda.fma.int32.int32.int32")
cuda_fma_int8_int8_int32 = visa.scalar_multiply_add(
    2.5, "int8", "int8", "int32", name="cuda.fma.int8.int8.int32")
cuda_fma_int8_int8_int8 = visa.scalar_multiply_add(
    2.5, "int8", "int8", "int8", name="cuda.fma.int8.int8.int8")


all_scalar_isa = [
    cuda_add_fp16_fp16_fp16,
    cuda_add_fp16_fp16_fp32,
    cuda_add_fp32_fp32_fp32,
    cuda_add_int32_int32_int32,
    cuda_add_int8_int8_int32,
    cuda_add_int8_int8_int8,

    cuda_sub_fp16_fp16_fp16,
    cuda_sub_fp16_fp16_fp32,
    cuda_sub_fp32_fp32_fp32,
    cuda_sub_int32_int32_int32,
    cuda_sub_int8_int8_int32,
    cuda_sub_int8_int8_int8,

    cuda_mul_fp16_fp16_fp16,
    cuda_mul_fp16_fp16_fp32,
    cuda_mul_fp32_fp32_fp32,
    cuda_mul_int32_int32_int32,
    cuda_mul_int8_int8_int32,
    cuda_mul_int8_int8_int8,

    cuda_div_fp16_fp16_fp16,
    cuda_div_fp16_fp16_fp32,
    cuda_div_fp32_fp32_fp32,
    cuda_div_int32_int32_int32,
    cuda_div_int8_int8_int32,
    cuda_div_int8_int8_int8,

    cuda_mod_fp16_fp16_fp16,
    cuda_mod_fp16_fp16_fp32,
    cuda_mod_fp32_fp32_fp32,
    cuda_mod_int32_int32_int32,
    cuda_mod_int8_int8_int32,
    cuda_mod_int8_int8_int8,

    cuda_fma_fp16_fp16_fp16,
    cuda_fma_fp16_fp16_fp32,
    cuda_fma_fp32_fp32_fp32,
    cuda_fma_int32_int32_int32,
    cuda_fma_int8_int8_int32,
    cuda_fma_int8_int8_int8,
]


def get_gemm_compute(m=16, n=16, k=16,
                     lhs_dtype="float16", rhs_dtype="float16", res_dtype="float32",
                     left_row_major=True, right_row_major=True, result_row_major=True):
    if left_row_major:
        left = tvm.te.placeholder([m, k], dtype=lhs_dtype, name="left_matrix")
    else:
        left = tvm.te.placeholder([k, m], dtype=lhs_dtype, name="left_matrix")

    if right_row_major:
        right = tvm.te.placeholder(
            [k, n], dtype=rhs_dtype, name="right_matrix")
    else:
        right = tvm.te.placeholder(
            [n, k], dtype=rhs_dtype, name="right_matrix")

    rk = tvm.te.reduce_axis([0, k], "rk")
    if left_row_major and right_row_major and result_row_major:
        result = tvm.te.compute(
            [m, n], lambda i, j: tvm.te.sum(
                left[i, rk].astype(res_dtype) * right[rk, j].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif left_row_major and right_row_major and not result_row_major:
        result = tvm.te.compute(
            [n, m], lambda i, j: tvm.te.sum(
                left[i, rk].astype(res_dtype) * right[rk, j].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif left_row_major and not right_row_major and result_row_major:
        result = tvm.te.compute(
            [m, n], lambda i, j: tvm.te.sum(
                left[i, rk].astype(res_dtype) * right[j, rk].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif left_row_major and not right_row_major and not result_row_major:
        result = tvm.te.compute(
            [n, m], lambda i, j: tvm.te.sum(
                left[i, rk].astype(res_dtype) * right[j, rk].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif not left_row_major and right_row_major and result_row_major:
        result = tvm.te.compute(
            [m, n], lambda i, j: tvm.te.sum(
                left[rk, i].astype(res_dtype) * right[rk, j].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif not left_row_major and right_row_major and not result_row_major:
        result = tvm.te.compute(
            [n, m], lambda i, j: tvm.te.sum(
                left[rk, i].astype(res_dtype) * right[rk, j].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    elif not left_row_major and not right_row_major and result_row_major:
        result = tvm.te.compute(
            [m, n], lambda i, j: tvm.te.sum(
                left[rk, i].astype(res_dtype) * right[j, rk].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    else:  # not left_row_major and not right_row_major and not result_row_major:
        result = tvm.te.compute(
            [n, m], lambda i, j: tvm.te.sum(
                left[rk, i].astype(res_dtype) * right[j, rk].astype(res_dtype), axis=[rk]), name="result_matrix"
        )
    return result.op


# matrix multiply-add
# fp16 fp16 fp32 row_major row_major
cuda_wmma_m16_n16_k16_row_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.row_major.fp16.fp16.fp32"
)
cuda_wmma_m8_n32_k16_row_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.row_major.fp16.fp16.fp32"
)
cuda_wmma_m32_n8_k16_row_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.row_major.fp16.fp16.fp32"
)
# fp16 fp16 fp32 row_major col_major
cuda_wmma_m16_n16_k16_row_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.col_major.fp16.fp16.fp32"
)
cuda_wmma_m8_n32_k16_row_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.col_major.fp16.fp16.fp32"
)
cuda_wmma_m32_n8_k16_row_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.col_major.fp16.fp16.fp32"
)
# fp16 fp16 fp32 col_major row_major
cuda_wmma_m16_n16_k16_col_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.col_major.row_major.fp16.fp16.fp32"
)
cuda_wmma_m8_n32_k16_col_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.col_major.row_major.fp16.fp16.fp32"
)
cuda_wmma_m32_n8_k16_col_major_row_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.col_major.row_major.fp16.fp16.fp32"
)
# fp16 fp16 fp32 col_major col_major
cuda_wmma_m16_n16_k16_col_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.col_major.col_major.fp16.fp16.fp32"
)
cuda_wmma_m8_n32_k16_col_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.col_major.col_major.fp16.fp16.fp32"
)
cuda_wmma_m32_n8_k16_col_major_col_major_fp16_fp16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float32", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.col_major.col_major.fp16.fp16.fp32"
)


# fp16 fp16 fp16 row_major row_major
cuda_wmma_m16_n16_k16_row_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.row_major.fp16.fp16.fp16"
)
cuda_wmma_m8_n32_k16_row_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.row_major.fp16.fp16.fp16"
)
cuda_wmma_m32_n8_k16_row_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.row_major.fp16.fp16.fp16"
)
# fp16 fp16 fp32 row_major col_major
cuda_wmma_m16_n16_k16_row_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.col_major.fp16.fp16.fp16"
)
cuda_wmma_m8_n32_k16_row_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.col_major.fp16.fp16.fp16"
)
cuda_wmma_m32_n8_k16_row_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=True, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.col_major.fp16.fp16.fp16"
)
# fp16 fp16 fp32 col_major row_major
cuda_wmma_m16_n16_k16_col_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.col_major.row_major.fp16.fp16.fp16"
)
cuda_wmma_m8_n32_k16_col_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.col_major.row_major.fp16.fp16.fp16"
)
cuda_wmma_m32_n8_k16_col_major_row_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.col_major.row_major.fp16.fp16.fp16"
)
# fp16 fp16 fp32 col_major col_major
cuda_wmma_m16_n16_k16_col_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.col_major.col_major.fp16.fp16.fp16"
)
cuda_wmma_m8_n32_k16_col_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.col_major.col_major.fp16.fp16.fp16"
)
cuda_wmma_m32_n8_k16_col_major_col_major_fp16_fp16_fp16 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="float16", rhs_dtype="float16",
                           res_dtype="float16", left_row_major=False, right_row_major=False, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.col_major.col_major.fp16.fp16.fp16"
)


wmma_sm70_isa = [
    cuda_wmma_m16_n16_k16_row_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m8_n32_k16_row_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m32_n8_k16_row_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m16_n16_k16_row_major_col_major_fp16_fp16_fp32,
    cuda_wmma_m8_n32_k16_row_major_col_major_fp16_fp16_fp32,
    cuda_wmma_m32_n8_k16_row_major_col_major_fp16_fp16_fp32,
    cuda_wmma_m16_n16_k16_col_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m8_n32_k16_col_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m32_n8_k16_col_major_row_major_fp16_fp16_fp32,
    cuda_wmma_m16_n16_k16_col_major_col_major_fp16_fp16_fp32,
    cuda_wmma_m8_n32_k16_col_major_col_major_fp16_fp16_fp32,
    cuda_wmma_m32_n8_k16_col_major_col_major_fp16_fp16_fp32,

    cuda_wmma_m16_n16_k16_row_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m8_n32_k16_row_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m32_n8_k16_row_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m16_n16_k16_row_major_col_major_fp16_fp16_fp16,
    cuda_wmma_m8_n32_k16_row_major_col_major_fp16_fp16_fp16,
    cuda_wmma_m32_n8_k16_row_major_col_major_fp16_fp16_fp16,
    cuda_wmma_m16_n16_k16_col_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m8_n32_k16_col_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m32_n8_k16_col_major_row_major_fp16_fp16_fp16,
    cuda_wmma_m16_n16_k16_col_major_col_major_fp16_fp16_fp16,
    cuda_wmma_m8_n32_k16_col_major_col_major_fp16_fp16_fp16,
    cuda_wmma_m32_n8_k16_col_major_col_major_fp16_fp16_fp16,
]


# int8 int8 int32 row_major row_major
cuda_wmma_m16_n16_k16_row_major_row_major_int8_int8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="int8", rhs_dtype="int8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.row_major.int8.int8.int32"
)
cuda_wmma_m8_n32_k16_row_major_row_major_int8_int8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="int8", rhs_dtype="int8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.row_major.int8.int8.int32"
)
cuda_wmma_m32_n8_k16_row_major_row_major_int8_int8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="int8", rhs_dtype="int8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.row_major.int8.int8.int32"
)

# uint8 uint8 int32 row_major row_major
cuda_wmma_m16_n16_k16_row_major_row_major_uint8_uint8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype="uint8", rhs_dtype="uint8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.row_major.uint8.uint8.int32"
)
cuda_wmma_m8_n32_k16_row_major_row_major_uint8_uint8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype="uint8", rhs_dtype="uint8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.row_major.uint8.uint8.int32"
)
cuda_wmma_m32_n8_k16_row_major_row_major_uint8_uint8_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype="uint8", rhs_dtype="uint8",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.row_major.uint8.uint8.int32"
)

# bf16 bf16 fp32 row_major row_major
cuda_wmma_m16_n16_k16_row_major_row_major_bf16_bf16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=16, lhs_dtype=f"custom[{cuda_bf16}]", rhs_dtype=f"custom[{cuda_bf16}]",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k16.row_major.row_major.bf16.bf16.fp32"
)
cuda_wmma_m8_n32_k16_row_major_row_major_bf16_bf16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=32, k=16, lhs_dtype=f"custom[{cuda_bf16}]", rhs_dtype=f"custom[{cuda_bf16}]",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n32.k16.row_major.row_major.bf16.bf16.fp32"
)
cuda_wmma_m32_n8_k16_row_major_row_major_bf16_bf16_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=32, n=8, k=16, lhs_dtype=f"custom[{cuda_bf16}]", rhs_dtype=f"custom[{cuda_bf16}]",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m32.n8.k16.row_major.row_major.bf16.bf16.fp32"
)

# tf32 tf32 fp32 row_major row_major
cuda_wmma_m16_n16_k8_row_major_row_major_tf32_tf32_fp32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=16, n=16, k=8, lhs_dtype=f"custom[{cuda_tf32}]", rhs_dtype=f"custom[{cuda_tf32}]",
                           res_dtype="float32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m16.n16.k8.row_major.row_major.tf32.tf32.fp32"
)

# fp64 fp64 fp64 row_major row_major
cuda_wmma_m8_n8_k4_row_major_row_major_fp64_fp64_fp64 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=8, k=4, lhs_dtype="float64", rhs_dtype="float64",
                           res_dtype="float64", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n8.k4.row_major.row_major.fp64.fp64.fp64"
)

# int4 int4 int32 row_major row_major
cuda_wmma_m8_n8_k32_row_major_row_major_int4_int4_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=8, k=32, lhs_dtype="int4", rhs_dtype="int4",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n8.k32.row_major.row_major.int4.int4.int32"
)

# uint4 uint4 int32 row_major row_major
cuda_wmma_m8_n8_k32_row_major_row_major_uint4_uint4_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=8, k=32, lhs_dtype="uint4", rhs_dtype="uint4",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n8.k32.row_major.row_major.uint4.uint4.int32"
)

# bin1 bin1 int32 row_major row_major
cuda_wmma_m8_n8_k128_row_major_row_major_bin_bin_int32 = visa.matrix_isa(
    64.0, get_gemm_compute(m=8, n=8, k=128, lhs_dtype="uint1", rhs_dtype="uint1",
                           res_dtype="int32", left_row_major=True, right_row_major=True, result_row_major=True),
    name="cuda.wmma.m8.n8.k128.row_major.row_major.bin.bin.int32"
)


wmma_sm75_isa = [
    *wmma_sm70_isa,
    cuda_wmma_m16_n16_k16_row_major_row_major_int8_int8_int32,
    cuda_wmma_m8_n32_k16_row_major_row_major_int8_int8_int32,
    cuda_wmma_m32_n8_k16_row_major_row_major_int8_int8_int32,
    cuda_wmma_m16_n16_k16_row_major_row_major_uint8_uint8_int32,
    cuda_wmma_m8_n32_k16_row_major_row_major_uint8_uint8_int32,
    cuda_wmma_m32_n8_k16_row_major_row_major_uint8_uint8_int32,
    cuda_wmma_m8_n8_k32_row_major_row_major_int4_int4_int32,
    cuda_wmma_m8_n8_k32_row_major_row_major_uint4_uint4_int32,
    cuda_wmma_m8_n8_k128_row_major_row_major_bin_bin_int32
]


wmma_sm80_isa = [
    *wmma_sm75_isa,
    cuda_wmma_m16_n16_k16_row_major_row_major_bf16_bf16_fp32,
    cuda_wmma_m8_n32_k16_row_major_row_major_bf16_bf16_fp32,
    cuda_wmma_m32_n8_k16_row_major_row_major_bf16_bf16_fp32,
    cuda_wmma_m16_n16_k8_row_major_row_major_tf32_tf32_fp32,
    cuda_wmma_m8_n8_k4_row_major_row_major_fp64_fp64_fp64
]
