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

#include <auto_compute/state.h>
#include <auto_tensorize/pattern.h>

using namespace tvm;
namespace ditto {
using namespace ditto::auto_compute;
namespace auto_tensorize {

using Intrinsic = te::TensorIntrin;

/*!
 * \brief A class for capsule intrinsic.
 */
class PackedIntrinsicNode : public Object {
public:
  /*! \brief The load intrinsics */
  Array<Intrinsic> load_intrinsics;
  /*! \brief The compute intrinsic */
  Intrinsic compute_intrinsic;
  /*! \brief The store intrinsic */
  Intrinsic store_intrinsic;
  /*! \brief The scopes of input data */
  Array<String> load_scopes;
  /*! \brief The scope of compute result */
  String compute_scope;
  /*! \brief The scope of output */
  String store_scope;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("load_intrinsics", &load_intrinsics);
    v->Visit("compute_intrinsic", &compute_intrinsic);
    v->Visit("store_intrinsic", &store_intrinsic);
    v->Visit("load_scopes", &load_scopes);
    v->Visit("compute_scope", &compute_scope);
    v->Visit("store_scope", &store_scope);
  }

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.PackedIntrinsic";
  TVM_DECLARE_BASE_OBJECT_INFO(PackedIntrinsicNode, Object);
};

class PackedIntrinsic : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param load_intrinsics Load intrinsics
   * \param compute_intrinsic Compute intrinsic
   * \param store_intrinsic Store intrinsic
   * \param load_scopes Load scope
   * \param compute_scope Compute scope
   * \param store_scope Store scope
   */
  TVM_DLL PackedIntrinsic(Array<Intrinsic> load_intrinsics,
                          Intrinsic compute_intrinsic,
                          Intrinsic store_intrinsic, Array<String> load_scopes,
                          String compute_scope, String store_scope);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PackedIntrinsic, ObjectRef,
                                        PackedIntrinsicNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PackedIntrinsicNode);
};

} // namespace auto_tensorize

} // namespace ditto