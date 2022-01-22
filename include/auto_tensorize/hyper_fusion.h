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
#include <auto_tensorize/intrinsic.h>

using namespace tvm;
namespace ditto {
using namespace ditto::auto_compute;
namespace auto_tensorize {

/*!
 * \brief A class for fusion choice.
 */
class FusionChoiceNode : public Object {
public:
  /*! \brief The first cubic op */
  te::Operation first_op;
  /*! \brief The second cubic op */
  te::Operation second_op;
  /*! \brief The ordered iters from the second op */
  Array<tir::IterVar> ordered_iters;
  /*! \brief The attach postion for compute_at */
  int attach_pos;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("first_op", &first_op);
    v->Visit("second_op", &second_op);
    v->Visit("ordered_iters", &ordered_iters);
    v->Visit("attach_pos", &attach_pos);
  }

  static constexpr const char *_type_key = "ditto.auto_tensorize.FusionChoice";
  TVM_DECLARE_BASE_OBJECT_INFO(FusionChoiceNode, Object);
};

class FusionChoice : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param first_op The first cubic operation
   * \param second_op The second cubic operation
   * \param ordered_iters The ordered iters of the second op
   * \param attach_pos The position to compute_at
   */
  TVM_DLL FusionChoice(te::Operation first_op, te::Operation second_op,
                       Array<tir::IterVar> ordered_iters, int attach_pos);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FusionChoice, ObjectRef,
                                        FusionChoiceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FusionChoiceNode);
};

/*!
 * \brief A class for tensorize match info.
 */
class MatchInfoNode : public Object {
public:
  /*! \brief The axis to tensorize */
  Array<tir::IterVar> axis;
  /*! \brief The intrinsic to use */
  PackedIntrinsic intrin;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("axis", &axis);
    v->Visit("intrin", &intrin);
  }

  static constexpr const char *_type_key = "ditto.auto_tensorize.MatchInfo";
  TVM_DECLARE_BASE_OBJECT_INFO(MatchInfoNode, Object);
};

class MatchInfo : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param axis The axis to tensorize
   * \param intrin The intrinsic to use
   */
  TVM_DLL MatchInfo(Array<tir::IterVar> axis, PackedIntrinsic intrin);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MatchInfo, ObjectRef, MatchInfoNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchInfoNode);
};

/*!
 * \brief A class for tensorize a layer.
 */
class TensorizeHyperFusionStateNode : public Object {
public:
  /*! \brief The original layer */
  Layer layer;
  /*! \brief The first cubic op */
  te::Operation first_op;
  /*! \brief The second cubic op */
  te::Operation second_op;
  /*! \brief The prologues of the first op */
  Array<Array<te::Operation>> first_op_prologue;
  /*! \brief The prologues of the second op */
  Array<Array<te::Operation>> second_op_prologue;
  /*! \brief The iintermediate path between first op and second op */
  Array<te::Operation> inter_path;
  /*! \brief The epilogue */
  Array<te::Operation> epilogue;
  /*! \brief The spatial outer iters for fusion */
  Array<tir::IterVar> fused_spatial_outer_iters;
  /*! \brief The reduce outer iters for fusion */
  Array<tir::IterVar> fused_reduce_outer_iters;
  /*! \brief The iters to be tensorized */
  Map<te::Operation, Array<tir::IterVar>> tensorize_iters;
  /*! \brief The intrinsic to be used in tensorization */
  Map<te::Operation, PackedIntrinsic> tensorize_intrinsics;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layer", &layer);
    v->Visit("first_op", &first_op);
    v->Visit("second_op", &second_op);
    v->Visit("first_op_prologue", &first_op_prologue);
    v->Visit("second_op_prologue", &second_op_prologue);
    v->Visit("inter_path", &inter_path);
    v->Visit("epilogue", &epilogue);
    v->Visit("fused_spatial_outer_iters", &fused_spatial_outer_iters);
    v->Visit("fused_reduce_outer_iters", &fused_reduce_outer_iters);
    v->Visit("tensorize_iters", &tensorize_iters);
    v->Visit("tensorize_intrinsics", &tensorize_intrinsics);
  }

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.TensorizeHyperFusionState";
  TVM_DECLARE_BASE_OBJECT_INFO(TensorizeHyperFusionStateNode, Object);
};

class TensorizeHyperFusionState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param layer The layer
   * \param fuse_choice The fusion choice
   * \param match_info The tensorize matching info
   */
  TVM_DLL TensorizeHyperFusionState(Layer layer, FusionChoice fuse_choice,
                                    Map<te::Operation, MatchInfo> match_info);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorizeHyperFusionState, ObjectRef,
                                        TensorizeHyperFusionStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorizeHyperFusionStateNode);
};

} // namespace auto_tensorize

} // namespace ditto