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
 * \brief A base class for memory pattern.
 */
class PatternNode : public Object {
public:
  /*! \brief The name of this pattern */
  String name;
  /*! \brief The grain of this memory */
  te::Tensor grain;
  /*! \brief The qualifier of this pattern */
  String qualifier;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("grain", &grain);
    v->Visit("qualifier", &qualifier);
  }

  static constexpr const char *_type_key = "ditto.hardware.Pattern";
  TVM_DECLARE_BASE_OBJECT_INFO(PatternNode, Object);
};

class Pattern : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the pattern
   * \param grain The grain of the pattern
   * \param qualifier The qualifier of the pattern
   */
  TVM_DLL Pattern(String name, te::Tensor grain, String qualifier);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Pattern, ObjectRef, PatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PatternNode);
};

} // namespace hardware

} // namespace ditto