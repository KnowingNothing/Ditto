#pragma once

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware parameters.
 */
class HardwareParamNode : public Object {
public:
  static constexpr const char *_type_key = "ditto.hardware.HardwareParam";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareParamNode, Object);
};

class HardwareParam : public ObjectRef {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareParam, ObjectRef,
                                        HardwareParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareParamNode);
};

} // namespace hardware

} // namespace ditto