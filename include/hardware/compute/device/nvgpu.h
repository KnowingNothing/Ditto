#pragma once

#include <hardware/base/hw_param.h>
#include <hardware/compute/device/hw_device.h>
#include <hardware/compute/group/homo_group.h>
#include <hardware/compute/processor/hetero_processor.h>
#include <hardware/pattern/matrix_pattern.h>
#include <hardware/pattern/scalar_pattern.h>
#include <hardware/visa/scalar_isa.h>

namespace ditto {

namespace hardware {

static const std::unordered_map<std::string, double> NvGPULatencyTable = {
    // TODO: measure the real latency of these instructions
    /* binary add */
    {"cuda.add.fp32.fp32.fp32", 1.0},
    {"cuda.add.fp16.fp16.fp32", 1.0},
    {"cuda.add.fp16.fp16.fp16", 1.0},
    {"cuda.add.int32.int32.int32", 1.0},
    {"cuda.add.int8.int8.int32", 1.0},
    {"cuda.add.int8.int8.int8", 1.0},

    /* binary sub */
    {"cuda.sub.fp32.fp32.fp32", 1.0},
    {"cuda.sub.fp16.fp16.fp32", 1.0},
    {"cuda.sub.fp16.fp16.fp16", 1.0},
    {"cuda.sub.int32.int32.int32", 1.0},
    {"cuda.sub.int8.int8.int32", 1.0},
    {"cuda.sub.int8.int8.int8", 1.0},

    /* binary mul */
    {"cuda.mul.fp32.fp32.fp32", 2.0},
    {"cuda.mul.fp16.fp16.fp32", 2.0},
    {"cuda.mul.fp16.fp16.fp16", 2.0},
    {"cuda.mul.int32.int32.int32", 2.0},
    {"cuda.mul.int8.int8.int32", 2.0},
    {"cuda.mul.int8.int8.int8", 2.0},

    /* binary div */
    {"cuda.div.fp32.fp32.fp32", 2.0},
    {"cuda.div.fp16.fp16.fp32", 2.0},
    {"cuda.div.fp16.fp16.fp16", 2.0},
    {"cuda.div.int32.int32.int32", 2.0},
    {"cuda.div.int8.int8.int32", 2.0},
    {"cuda.div.int8.int8.int8", 2.0},

    /* binary mod */
    {"cuda.mod.fp32.fp32.fp32", 2.0},
    {"cuda.mod.fp16.fp16.fp32", 2.0},
    {"cuda.mod.fp16.fp16.fp16", 2.0},
    {"cuda.mod.int32.int32.int32", 2.0},
    {"cuda.mod.int8.int8.int32", 2.0},
    {"cuda.mod.int8.int8.int8", 2.0},

    /* multiply-add */
    {"cuda.fma.fp32.fp32.fp32", 2.5},
    {"cuda.fma.fp16.fp16.fp32", 2.5},
    {"cuda.fma.fp16.fp16.fp16", 2.5},
    {"cuda.fma.int32.int32.int32", 2.5},
    {"cuda.fma.int8.int8.int32", 2.5},
    {"cuda.fma.int8.int8.int8", 2.5},

    /* matrix multiply-add */
    // fp16 fp16 fp32 row_major row_major
    {"cuda.wmma.m16.n16.k16.row_major.row_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.row_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.row_major.fp16.fp16.fp32", 64.0},
    // fp16 fp16 fp32 row_major col_major
    {"cuda.wmma.m16.n16.k16.row_major.col_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.col_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.col_major.fp16.fp16.fp32", 64.0},
    // fp16 fp16 fp32 col_major row_major
    {"cuda.wmma.m16.n16.k16.col_major.row_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m8.n32.k16.col_major.row_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m32.n8.k16.col_major.row_major.fp16.fp16.fp32", 64.0},
    // fp16 fp16 fp32 col_major col_major
    {"cuda.wmma.m16.n16.k16.col_major.col_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m8.n32.k16.col_major.col_major.fp16.fp16.fp32", 64.0},
    {"cuda.wmma.m32.n8.k16.col_major.col_major.fp16.fp16.fp32", 64.0},

    // fp16 fp16 fp16 row_major row_major
    {"cuda.wmma.m16.n16.k16.row_major.row_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.row_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.row_major.fp16.fp16.fp16", 64.0},
    // fp16 fp16 fp16 row_major col_major
    {"cuda.wmma.m16.n16.k16.row_major.col_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.col_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.col_major.fp16.fp16.fp16", 64.0},
    // fp16 fp16 fp16 col_major row_major
    {"cuda.wmma.m16.n16.k16.col_major.row_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m8.n32.k16.col_major.row_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m32.n8.k16.col_major.row_major.fp16.fp16.fp16", 64.0},
    // fp16 fp16 fp16 col_major col_major
    {"cuda.wmma.m16.n16.k16.col_major.col_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m8.n32.k16.col_major.col_major.fp16.fp16.fp16", 64.0},
    {"cuda.wmma.m32.n8.k16.col_major.col_major.fp16.fp16.fp16", 64.0},

    // int8 int8 int32 row_major row_major
    {"cuda.wmma.m16.n16.k16.row_major.row_major.int8.int8.int32", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.row_major.int8.int8.int32", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.row_major.int8.int8.int32", 64.0},

    // uint8 uint8 int32 row_major row_major
    {"cuda.wmma.m16.n16.k16.row_major.row_major.uint8.uint8.int32", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.row_major.uint8.uint8.int32", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.row_major.uint8.uint8.int32", 64.0},

    // bf16 bf16 fp32 row_major row_major
    {"cuda.wmma.m16.n16.k16.row_major.row_major.bf16.bf16.fp32", 64.0},
    {"cuda.wmma.m8.n32.k16.row_major.row_major.bf16.bf16.fp32", 64.0},
    {"cuda.wmma.m32.n8.k16.row_major.row_major.bf16.bf16.fp32", 64.0},

    // tf32 tf32 fp32 row_major row_major
    {"cuda.wmma.m16.n16.k8.row_major.row_major.tf32.tf32.fp32", 64.0},

    // fp64 fp64 fp64 row_major row_major
    {"cuda.wmma.m8.n8.k4.row_major.row_major.fp64.fp64.fp64", 64.0},

    // int4 int4 int32 row_major row_major
    {"cuda.wmma.m8.n8.k32.row_major.row_major.int4.int4.int32", 64.0},

    // uint4 uint4 int32 row_major row_major
    {"cuda.wmma.m8.n8.k32.row_major.row_major.uint4.uint4.int32", 64.0},

    // bin1 bin1 int32 row_major row_major
    {"cuda.wmma.m8.n8.k128.row_major.row_major.bin1.bin1.int32", 64.0},

};

/*!
 * \brief A base class for NvGPU params.
 */
class NvGPUParamNode : public HardwareParamNode {
public:
  int GetNum32BitRegistersPerSM(String sm_arch) {
    if (sm_arch == "sm61") {
      return 65536;
    } else {
      CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
      return -1;
    }
  }

  int GetNumProcessorPerSM(String sm_arch) {
    if (sm_arch == "sm61") {
      return 2;
    } else {
      CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
      return -1;
    }
  }

  int GetNumSMPerGPU(String sm_arch) {
    if (sm_arch == "sm61") {
      return 56;
    } else {
      CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
      return -1;
    }
  }

  double GetKBSharedMemPerSM(String sm_arch) {
    if (sm_arch == "sm61") {
      return 64.0;
    } else {
      CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
      return -1;
    }
  }

  double GetGBGlobalMemPerSM(String sm_arch) {
    if (sm_arch == "sm61") {
      return 16.0;
    } else {
      CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
      return -1;
    }
  }

  static constexpr const char *_type_key = "ditto.hardware.NvGPUParam";
  TVM_DECLARE_FINAL_OBJECT_INFO(NvGPUParamNode, HardwareParamNode);
};

class NvGPUParam : public HardwareParam {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NvGPUParam, HardwareParam,
                                        NvGPUParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NvGPUParamNode);
};

HardwareDevice NvSimtGPU(String sm_arch);

} // namespace hardware

} // namespace ditto