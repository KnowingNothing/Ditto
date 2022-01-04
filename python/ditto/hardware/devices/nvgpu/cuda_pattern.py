from .cuda_dtype import cuda_bf16, cuda_tf32
from ... import pattern

scalar_patterns = {
    "scalar_fp16": pattern.scalar_pattern(
        "float16", "", name="scalar_fp16"),
    "scalar_fp32": pattern.scalar_pattern(
        "float32", "", name="scalar_fp32"),
    "scalar_int32": pattern.scalar_pattern(
        "int32", "", name="scalar_int32"),
    "scalar_int8": pattern.scalar_pattern(
        "int8", "", name="scalar_int8"),
}


wmma_sm70_patterns = {
    # fp16 matrix a
    "matrix_a_16x16_row_major_fp16": pattern.matrix_pattern(
        "float16", 16, 16, "matrix_a::row_major", name="matrix_a_16x16_row_major_fp16"
    ),
    "matrix_a_16x16_col_major_fp16": pattern.matrix_pattern(
        "float16", 16, 16, "matrix_a::col_major", name="matrix_a_16x16_col_major_fp16"
    ),
    "matrix_a_8x16_row_major_fp16": pattern.matrix_pattern(
        "float16", 8, 16, "matrix_a::row_major", name="matrix_a_8x16_row_major_fp16"
    ),
    "matrix_a_8x16_col_major_fp16": pattern.matrix_pattern(
        "float16", 16, 8, "matrix_a::col_major", name="matrix_a_8x16_col_major_fp16"
    ),
    "matrix_a_32x16_row_major_fp16": pattern.matrix_pattern(
        "float16", 32, 16, "matrix_a::row_major", name="matrix_a_32x16_row_major_fp16"
    ),
    "matrix_a_32x16_col_major_fp16": pattern.matrix_pattern(
        "float16", 16, 32, "matrix_a::col_major", name="matrix_a_32x16_col_major_fp16"
    ),
    # fp16 matrix b
    "matrix_b_16x16_row_major_fp16": pattern.matrix_pattern(
        "float16", 16, 16, "matrix_b::row_major", name="matrix_b_16x16_row_major_fp16"
    ),
    "matrix_b_16x16_col_major_fp16": pattern.matrix_pattern(
        "float16", 16, 16, "matrix_b::col_major", name="matrix_b_16x16_col_major_fp16"
    ),
    "matrix_b_16x32_row_major_fp16": pattern.matrix_pattern(
        "float16", 16, 32, "matrix_b::row_major", name="matrix_b_16x32_row_major_fp16"
    ),
    "matrix_b_16x32_col_major_fp16": pattern.matrix_pattern(
        "float16", 32, 16, "matrix_b::col_major", name="matrix_b_16x32_col_major_fp16"
    ),
    "matrix_b_16x8_row_major_fp16": pattern.matrix_pattern(
        "float16", 16, 8, "matrix_b::row_major", name="matrix_b_16x8_row_major_fp16"
    ),
    "matrix_b_16x8_col_major_fp16": pattern.matrix_pattern(
        "float16", 8, 16, "matrix_b::col_major", name="matrix_b_16x8_col_major_fp16"
    ),
    # fp16 accumulator
    "accumulator_16x16_row_major_fp16": pattern.matrix_pattern(
        "float16", 16, 16, "accumulator::row_major", name="accumulator_16x16_row_major_fp16"
    ),
    "accumulator_32x8_row_major_fp16": pattern.matrix_pattern(
        "float16", 32, 8, "accumulator::row_major", name="accumulator_32x8_row_major_fp16"
    ),
    "accumulator_8x32_row_major_fp16": pattern.matrix_pattern(
        "float16", 8, 32, "accumulator::row_major", name="accumulator_8x32_row_major_fp16"
    ),
    # fp32 accumulator
    "accumulator_16x16_row_major_fp32": pattern.matrix_pattern(
        "float32", 16, 16, "accumulator::row_major", name="accumulator_16x16_row_major_fp32"
    ),
    "accumulator_32x8_row_major_fp32": pattern.matrix_pattern(
        "float32", 32, 8, "accumulator::row_major", name="accumulator_32x8_row_major_fp32"
    ),
    "accumulator_8x32_row_major_fp32": pattern.matrix_pattern(
        "float32", 8, 32, "accumulator::row_major", name="accumulator_8x32_row_major_fp32"
    ),
}


wmma_sm75_patterns = {
    **wmma_sm70_patterns,
    # int8 matrix a
    "matrix_a_16x16_row_major_int8": pattern.matrix_pattern(
        "int8", 16, 16, "matrix_a::row_major", name="matrix_a_16x16_row_major_int8"
    ),
    "matrix_a_8x16_row_major_int8": pattern.matrix_pattern(
        "int32", 8, 16, "matrix_a::row_major", name="matrix_a_8x16_row_major_int8"
    ),
    "matrix_a_32x16_row_major_int8": pattern.matrix_pattern(
        "int8", 32, 16, "matrix_a::row_major", name="matrix_a_32x16_row_major_int8"
    ),
    # int8 matrix b
    "matrix_b_16x16_row_major_int8": pattern.matrix_pattern(
        "int8", 16, 16, "matrix_b::row_major", name="matrix_b_16x16_row_major_int8"
    ),
    "matrix_b_16x8_row_major_int8": pattern.matrix_pattern(
        "int8", 16, 8, "matrix_b::row_major", name="matrix_b_16x8_row_major_int8"
    ),
    "matrix_b_16x32_row_major_int8": pattern.matrix_pattern(
        "int8", 16, 32, "matrix_b::row_major", name="matrix_b_16x32_row_major_int8"
    ),
    # uint8 matrix a
    "matrix_a_16x16_row_major_uint8": pattern.matrix_pattern(
        "uint8", 16, 16, "matrix_a::row_major", name="matrix_a_16x16_row_major_uint8"
    ),
    "matrix_a_8x16_row_major_uint8": pattern.matrix_pattern(
        "uint32", 8, 16, "matrix_a::row_major", name="matrix_a_8x16_row_major_uint8"
    ),
    "matrix_a_32x16_row_major_uint8": pattern.matrix_pattern(
        "uint8", 32, 16, "matrix_a::row_major", name="matrix_a_32x16_row_major_uint8"
    ),
    # uint8 matrix b
    "matrix_b_16x16_row_major_uint8": pattern.matrix_pattern(
        "uint8", 16, 16, "matrix_b::row_major", name="matrix_b_16x16_row_major_uint8"
    ),
    "matrix_b_16x8_row_major_uint8": pattern.matrix_pattern(
        "uint8", 16, 8, "matrix_b::row_major", name="matrix_b_16x8_row_major_uint8"
    ),
    "matrix_b_16x32_row_major_uint8": pattern.matrix_pattern(
        "uint8", 16, 32, "matrix_b::row_major", name="matrix_b_16x32_row_major_uint8"
    ),
    # int4 matrix a
    "matrix_a_8x32_row_major_int4": pattern.matrix_pattern(
        "int4", 8, 32, "matrix_a::row_major", name="matrix_a_8x32_row_major_int4"
    ),
    # int4 matrix b
    "matrix_b_32x8_row_major_int4": pattern.matrix_pattern(
        "int4", 32, 8, "matrix_b::row_major", name="matrix_b_32x8_row_major_int4"
    ),
    # uint4 matrix a
    "matrix_a_8x32_row_major_uint4": pattern.matrix_pattern(
        "uint4", 8, 32, "matrix_a::row_major", name="matrix_a_8x32_row_major_uint4"
    ),
    # uint4 matrix b
    "matrix_b_32x8_row_major_uint4": pattern.matrix_pattern(
        "uint4", 32, 8, "matrix_b::row_major", name="matrix_b_32x8_row_major_uint4"
    ),
    # bin matrix a
    "matrix_a_8x128_row_major_bin": pattern.matrix_pattern(
        "uint1", 8, 128, "matrix_a::row_major", name="matrix_a_8x128_row_major_bin"
    ),
    # bin matrix b
    "matrix_b_128x8_row_major_bin": pattern.matrix_pattern(
        "uint1", 128, 8, "matrix_b::row_major", name="matrix_b_128x8_row_major_bin"
    ),
    # int32 accumulator
    "accumulator_16x16_row_major_int32": pattern.matrix_pattern(
        "int32", 16, 16, "accumulator::row_major", name="accumulator_16x16_row_major_int32"
    ),
    "accumulator_32x8_row_major_int32": pattern.matrix_pattern(
        "int32", 32, 8, "accumulator::row_major", name="accumulator_32x8_row_major_int32"
    ),
    "accumulator_8x32_row_major_int32": pattern.matrix_pattern(
        "int32", 8, 32, "accumulator::row_major", name="accumulator_8x32_row_major_int32"
    ),
    "accumulator_8x8_row_major_int32": pattern.matrix_pattern(
        "int32", 8, 8, "accumulator::row_major", name="accumulator_8x8_row_major_int32"
    ),
}


wmma_sm80_patterns = {
    **wmma_sm75_patterns,
    # bf16 matrix a
    "matrix_a_16x16_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 16, 16, "matrix_a::row_major", name="matrix_a_16x16_row_major_bf16"
    ),
    "matrix_a_8x16_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 8, 16, "matrix_a::row_major", name="matrix_a_8x16_row_major_bf16"
    ),
    "matrix_a_32x16_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 32, 16, "matrix_a::row_major", name="matrix_a_32x16_row_major_bf16"
    ),
    # bf16 matrix b
    "matrix_b_16x16_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 16, 16, "matrix_b::row_major", name="matrix_b_16x16_row_major_bf16"
    ),
    "matrix_b_16x8_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 16, 8, "matrix_b::row_major", name="matrix_b_16x8_row_major_bf16"
    ),
    "matrix_b_16x32_row_major_bf16": pattern.matrix_pattern(
        f"custom[{cuda_bf16}]", 16, 32, "matrix_b::row_major", name="matrix_b_16x32_row_major_bf16"
    ),
    # tf32 matrix a
    "matrix_a_16x8_row_major_tf32": pattern.matrix_pattern(
        f"custom[{cuda_tf32}]", 16, 8, "matrix_a::row_major", name="matrix_a_16x16_row_major_tf32"
    ),
    # tf32 matrix b
    "matrix_b_8x16_row_major_tf32": pattern.matrix_pattern(
        f"custom[{cuda_tf32}]", 8, 16, "matrix_b::row_major", name="matrix_b_8x16_row_major_tf32"
    ),
    # fp64 matrix a
    "matrix_a_8x4_row_major_fp64": pattern.matrix_pattern(
        "float64", 8, 4, "matrix_a::row_major", name="matrix_a_8x4_row_major_fp64"
    ),
    # fp64 matrix b
    "matrix_b_4x8_row_major_fp64": pattern.matrix_pattern(
        "float64", 4, 8, "matrix_b::row_major", name="matrix_b_4x8_row_major_fp64"
    ),
    # fp64 accumulator
    "accumulator_8x8_row_major_fp64": pattern.matrix_pattern(
        "float64", 8, 8, "accumulator::row_major", name="accumulator_8x8_row_major_fp64"
    )
}
