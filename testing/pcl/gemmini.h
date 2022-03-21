// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemmini_params.h"

#define GEMMINI_ASSERTIONS

// Rounding right shift equation:
// https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#ifndef ELEM_T_IS_FLOAT
#define ROUNDING_RIGHT_SHIFT(x, shift)                                         \
  ({                                                                           \
    (shift) > 0                                                                \
        ? (((x) >> (shift)) +                                                  \
           (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) &                  \
            ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) |    \
             (((x) >> (shift)) & 1))))                                         \
        : ((x) << (-(shift)));                                                 \
  })
#else
#define ROUNDING_RIGHT_SHIFT(x, shift) ((x) / (1 << (shift)))
#endif

// Accelerator interface
#include "xcustom.h"

#define k_CONFIG 0
#define k_MVIN_INPUT 1
#define k_MVIN_FILTER 2
#define k_COMPUTE_PRELOADED 3
#define k_COMPUTE_ACCUMULATE 4
#define k_PRELOAD 6
#define k_FLUSH 7
#define k_MVOUT 8
#define k_CONV_PRE 3
#define k_CONV_EXEC 4
#define k_FENCE 10
#define k_CONFIG_PADDING_DATA 15
#define k_CONFIG_DATAFLOW 16
#define k_POSTPROC 17
#define k_CONFIG_LEAKY 18

#define k_CONFIG_EXEC 5
#define k_CONFIG_SLOT 6
#define k_CONFIG_ACTI 11
#define k_CONFIG_STRI 12
#define k_CONFIG_POOL_FUNC 13
#define k_CONFIG_POOL_PARA 14
#define k_CONFIG_SRAM_BYPASS 9
//#define k_LOOP_WS 8

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2

#define XCUSTOM_ACC 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
  union {
    elem_t_bits b;
    elem_t f;
  } un;

  un.b = x;
  return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
  union {
    elem_t_bits b;
    elem_t f;
  } un;

  un.f = x;
  return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
  union {
    acc_t_bits b;
    acc_t f;
  } un;

  un.b = x;
  return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
  union {
    acc_t_bits b;
    acc_t f;
  } un;

  un.f = x;
  return un.b;
}

bool elem_t_isnan(elem_t x) {
  elem_t_bits bits = elem_t_to_elem_t_bits(x);
  uint64_t exp =
      (bits >> (ELEM_T_SIG_BITS - 1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
  uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
  bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
  bool is_not_inf = sig != 0;
  return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
  acc_t_bits bits = acc_t_to_acc_t_bits(x);
  uint64_t exp =
      (bits >> (ACC_T_SIG_BITS - 1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
  uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
  bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
  bool is_not_inf = sig != 0;
  return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.b = x;
  return un.f;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}
#endif

#ifdef HAS_MVIN_ACC_SCALE
scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
  union {
    scale_acc_t_bits b;
    scale_acc_t f;
  } un;

  un.b = x;
  return un.f;
}

scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
  union {
    scale_acc_t_bits b;
    scale_acc_t f;
  } un;

  un.f = x;
  return un.b;
}
#endif

// WSSystolic accelerator instruction
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct)                           \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// config

// #define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
// #define systolic_config_execute(...) GET_MACRO(__VA_ARGS__,
// systolic_config_execute_bf, , systolic_config_execute, )(__VA_ARGS__)

//#define systolic_config_execute_p(padding_size,kernel_size , out_size) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,((uint64_t)(padding_size) << 8) | (uint64_t)(kernel_size), (uint64_t)(out_size), k_CONFIG_EXEC)

#define systolic_config_dataflow_type(dataflow_type)                           \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dataflow_type),             \
                           (uint64_t)(0), k_CONFIG_DATAFLOW)

#define systolic_config_execute(padding_size, kernel_size, out_size, ctrl,     \
                                sgn)                                           \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,                                        \
                           (uint64_t)(kernel_size) | ((uint64_t)ctrl << 32) |  \
                               ((uint64_t)sgn << 34) |                         \
                               ((uint64_t)(padding_size) << 8),                \
                           (uint64_t)(out_size), k_CONFIG_EXEC)

#define systolic_config_slot_size(input_slot_size, out_slot_size,              \
                                  filter_slot_size)                            \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,                                        \
                           ((uint64_t)(out_slot_size) << ADDR_LEN) |           \
                               (input_slot_size),                              \
                           (uint64_t)(filter_slot_size), k_CONFIG_SLOT)

#define systolic_config_activation(act_func, act_param)                        \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(act_func),                  \
                           (uint64_t)(act_param), k_CONFIG_ACTI)

#define systolic_padding_data(padding_left, padding_data)                      \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,                                        \
                           (uint64_t)(padding_data) |                          \
                               (uint64_t)((padding_left) << 16),               \
                           (uint64_t)(0), k_CONFIG_PADDING_DATA)

#define systolic_config_stride(conv_stride)                                    \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(conv_stride),               \
                           (uint64_t)(0), k_CONFIG_STRI)

#define systolic_pool_func(pool_func)                                          \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(pool_func), (uint64_t)(0),  \
                           k_CONFIG_POOL_FUNC)

#define systolic_pool_param(rs1, rs2)                                          \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(rs1), (uint64_t)(rs2),      \
                           k_CONFIG_POOL_PARA)

#define systolic_sram_cfg(rs1, rs2)                                            \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(rs1), (uint64_t)(rs2),      \
                           k_CONFIG_SRAM_BYPASS)

// sel_mem = 0: input buffer
// sel_mem = 1: result buffer
#define systolic_mvin_input(dram_addr, slot_id, sel_mem)                       \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr,                             \
                           ((uint64_t)(sel_mem) << ADDR_LEN) | (slot_id),      \
                           k_MVIN_INPUT)

#define systolic_mvin_filter(dram_addr, slot_id)                               \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr,                             \
                           ((uint64_t)(0) << ADDR_LEN) | (slot_id),            \
                           k_MVIN_FILTER)

// mv_out
#define systolic_mvout(dram_addr, slot_id, sel_mem)                            \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr,                             \
                           ((uint64_t)(sel_mem) << ADDR_LEN) | (slot_id),      \
                           k_MVOUT)

#define systolic_conv_pre(input_slot_id, input_sel_mem, filter_slot_id)        \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC, ((uint64_t)(input_sel_mem) << ADDR_LEN) | (input_slot_id),  \
      filter_slot_id, k_CONV_PRE)

#define systolic_conv_exec(out_slot_id, out_sel_mem, zeroize, zero_direction,  \
                           tiling)                                             \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(out_sel_mem) << ADDR_LEN) | (uint64_t)(out_slot_id),         \
      (uint64_t)(zeroize) | (uint64_t)(zero_direction) << 16 |                 \
          ((uint64_t)(tiling) << 32),                                          \
      k_CONV_EXEC)

#define systolic_postproc(reg_type, reg_addr, reg_data)                        \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,                                        \
                           (uint64_t)(reg_addr) | (uint64_t)(reg_type) << 8,   \
                           (uint64_t)(reg_data), k_POSTPROC)

#define systolic_postproc_offset(mutil_shift, out_offset)                      \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(3) << 8,                    \
                           (uint64_t)mutil_shift << 32 |                       \
                               (uint64_t)((uint32_t)(out_offset)),             \
                           k_POSTPROC)

#define systolic_leakyrelu(leakrelu_addr, leakrelu_data)                       \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(leakrelu_addr),             \
                           (uint64_t)(leakrelu_data), k_CONFIG_LEAKY)

#define systolic_fence() ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_FENCE)

// gemmini mv_in
#define gemmini_mvin(dram_addr, spad_addr)                                     \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvin(dram_addr, slot_id, cols, rows)                  \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr,                             \
                           ((uint64_t)(cols) << ADDR_LEN) | (slot_id), 0)

// gemmini mvout
#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows)               \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr,                             \
                           ((uint64_t)(rows) << (ADDR_LEN + 16)) |             \
                               ((uint64_t)(cols) << ADDR_LEN) |                \
                               (uint64_t)(spad_addr),                          \
                           k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr)                                    \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols,     \
                                           BD_rows)                            \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(A_rows) << (ADDR_LEN + 16)) |                                \
          ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A),                    \
      ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) |                               \
          ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD),                  \
      k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols,   \
                                             BD_rows)                          \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(A_rows) << (ADDR_LEN + 16)) |                                \
          ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A),                    \
      ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) |                               \
          ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD),                  \
      k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD)                                       \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD)                                     \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows)      \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) |                               \
          ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD),                  \
      ((uint64_t)(C_rows) << (ADDR_LEN + 16)) |                                \
          ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C),                    \
      k_PRELOAD)

#define gemmini_preload(BD, C)                                                 \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) gemmini_preload(GARBAGE_ADDR, C)

// weight-stationary matmul loop
// #define gemmini_loop_ws(A, B, I, J, K, bias) \
    // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(B) << 32) | (A),
// ((uint64_t)(bias) << 48) | ((uint64_t)(K) << 32) | ((J) << 16) | (I),
// k_LOOP_WS)

// config
#define gemmini_extended_config_ex(mode, act, sys_shift, acc_shift,            \
                                   relu6_shift, A_stride, A_transpose,         \
                                   B_transpose)                                \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(acc_shift) << 32) | ((uint64_t)(A_stride) << 16) |           \
          (B_transpose << 9) | (A_transpose << 8) | ((act) << 3) |             \
          ((mode) << 2) | CONFIG_EX,                                           \
      ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)

#define gemmini_config_ex(mode, act, sys_shift, acc_shift, relu6_shift)        \
  gemmini_extended_config_ex(mode, act, sys_shift, acc_shift, relu6_shift, 1,  \
                             0, 0)

#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
#define gemmini_extended_config_ld(stride, scale)                              \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | CONFIG_LD, stride,  \
      k_CONFIG)
#else
#define gemmini_extended_config_ld(stride, scale)                              \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)
#endif

#define gemmini_config_ld(stride)                                              \
  gemmini_extended_config_ld(stride, MVIN_SCALE_ONE)

#define gemmini_extended_config_st(stride, pool_stride, pool_size,             \
                                   pool_out_dim, porows, pocols, orows, ocols, \
                                   upad, lpad)                                 \
  ROCC_INSTRUCTION_RS1_RS2(                                                    \
      XCUSTOM_ACC,                                                             \
      ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) |                  \
          ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) |            \
          ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) |        \
          ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) |             \
          ((uint64_t)(pool_stride) << 4) | CONFIG_ST,                          \
      stride, k_CONFIG)

#define gemmini_config_st(stride)                                              \
  gemmini_extended_config_st(stride, 0, 0, 0, 0, 0, 0, 0, 0, 0)

// flush
#define gemmini_flush(skip)                                                    \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) ((x) * (scale))
#else
#define GEMMINI_SCALE(x, scale) x
#endif

#undef GEMMINI_SCALE

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {
  OS,
  WS,
  CPU
}; // TODO rename this so it's name also applies to convs

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H
