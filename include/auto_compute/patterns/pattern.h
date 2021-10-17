#pragma once

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

namespace ditto {
using namespace tvm;
namespace auto_compute {

/////////////////////////////////////
// Definitions for pattern
////////////////////////////////////

/*!
 * \brief PatternNode class.
 */
class PatternNode : public Object {
public:
  /*! \brief record tensors */
  Array<IntImm> tensor_ids;
  /*! \brief record matched iter vars */
  Array<Array<IntImm>> iter_ids_array;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("tensor_ids", &tensor_ids);
    v->Visit("iter_ids_array", &iter_ids_array);
  }

  static constexpr const char *_type_key = "ditto.auto_compute.Pattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternNode, Object);
};

class Pattern : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param tensor_ids The recorded ids of tensors
   * \param iter_ids_array The matched iter var ids
   */
  TVM_DLL Pattern(Array<IntImm> tensor_ids,
                  Array<Array<IntImm>> iter_ids_array);

  TVM_DEFINE_OBJECT_REF_METHODS(Pattern, ObjectRef, PatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PatternNode);
};

} // namespace auto_compute

} // namespace ditto