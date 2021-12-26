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
 * \brief A base class for hardware.
 */
class HardwareNode : public Object {
public:
  /*! \brief The name of this hardware */
  String name;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("name", &name); }

  static constexpr const char *_type_key = "ditto.hardware.Hardware";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareNode, Object);
};

class Hardware : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   */
  TVM_DLL Hardware(String name);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Hardware, ObjectRef, HardwareNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareNode);
};

} // namespace hardware

} // namespace ditto