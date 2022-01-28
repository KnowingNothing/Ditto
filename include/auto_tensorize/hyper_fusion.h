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
#include <unordered_set>
#include <vector>

#include <auto_compute/state.h>
#include <auto_tensorize/intrinsic.h>
#include <hardware/base/hw_param.h>

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

/*!
 * \brief A class for cuda tensorize context.
 */
class CUDATensorizeContextNode : public Object {
public:
  /*! \brief The layer to schedule */
  Layer layer;
  /*! \brief The tensorize and fusion state */
  TensorizeHyperFusionState state;
  /*! \brief The parameters of GPU */
  hardware::HardwareParam cuda_param;
  /*! \brief The second op's fragment */
  te::Tensor second_frag;
  /*! \brief The second op compute_at axis */
  tir::IterVar second_op_compute_axis;
  /*! \brief The inter path compute_at tensor */
  te::Tensor path_attach_tensor;
  /*! \brief The inter path compute_at axis */
  tir::IterVar path_attach_axis;
  /*! \brief The first prologue shared compute_at tensor */
  te::Tensor first_prologue_shared_attach_tensor;
  /*! \brief The first prologue shared compute_at axis */
  tir::IterVar first_prologue_shared_attach_axis;
  /*! \brief The second prologue frag compute_at tensor */
  te::Tensor first_frag_attach_tensor;
  /*! \brief The second prologue frag compute_at axis */
  tir::IterVar first_frag_attach_axis;
  /*! \brief The second prologue shared compute_at tensor */
  te::Tensor second_prologue_shared_attach_tensor;
  /*! \brief The second prologue shared compute_at axis */
  tir::IterVar second_prologue_shared_attach_axis;
  /*! \brief The second prologue frag compute_at tensor */
  te::Tensor second_frag_attach_tensor;
  /*! \brief The second prologue frag compute_at axis */
  tir::IterVar second_frag_attach_axis;
  /*! \brief ThreadIdx.z is used */
  bool tz_used{false};
  /*! \brief ThreadIdx.y is used */
  bool ty_used{false};

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layer", &layer);
    v->Visit("state", &state);
    v->Visit("cuda_param", &cuda_param);
  }
  /*!
   * \brief Check if has epilogue.
   */
  bool HasEpilogue();
  /*!
   * \brief Get the epilogue root op.
   */
  te::Operation EpilogueRootOp();
  /*!
   * \brief Get the non-root epilogue ops.
   */
  Array<te::Operation> EpilogueNonRootOps();
  /*!
   * \brief Check if has inter path.
   */
  bool HasInterPath();
  /*!
   * \brief Get the inter path root op.
   */
  te::Operation InterPathRootOp();
  /*!
   * \brief Get the non-root inter path ops.
   */
  Array<te::Operation> InterPathNonRootOps();
  /*!
   * \brief Split a dim into multi-parts from inner to outer.
   */
  Array<tir::IterVar> Split(te::Schedule sch, te::Operation op, tir::IterVar iv,
                            Array<PrimExpr> factors);
  /*!
   * \brief Fuse all the iter vars.
   */
  tir::IterVar FuseAll(te::Schedule sch, te::Operation op);
  /*!
   * \brief Fuse all the iter vars and split into multi-parts.
   */
  Array<tir::IterVar> FuseAllAndSplit(te::Schedule sch, te::Operation op,
                                      Array<PrimExpr> factors);
  /*!
   * \brief Compute inline the operation.
   */
  void Inline(te::Schedule sch, te::Operation op);
  /*!
   * \brief Check if an operation can be inlined.
   */
  bool CanInline(te::Operation op);
  /*!
   * \brief Get the outer and inner spatial axis index of the second op.
   */
  std::pair<std::vector<int>, std::vector<int>> SecondOpOuterInnerSpatialAxis();
  /*!
   * \brief Get the outer and inner reduce axis index of the second op.
   */
  std::pair<std::vector<int>, std::vector<int>> SecondOpOuterInnerReduceAxis();
  /*!
   * \brief Get the tensorized spatial axis index.
   */
  std::vector<int> TensorizeSpatialAxis(const te::Operation &op);
  /*!
   * \brief Get the tensorized reduce axis index.
   */
  std::vector<int> TensorizeReduceAxis(const te::Operation &op);
  /*!
   * \brief Check if the fusion and tensorize choices are valide.
   */
  bool ValidTensorizeFusion(const std::vector<int> &inner_index,
                            const std::vector<int> &tensorize_index);
  /*!
   * \brief Get the extents.
   */
  std::vector<int> GetSpatialExtentsByIndex(const te::Operation &op,
                                            const std::vector<int> &index);
  /*!
   * \brief Get the extents.
   */
  std::vector<int> GetReduceExtentsByIndex(const te::Operation &op,
                                           const std::vector<int> &index);
  /*!
   * \brief Check if the op is in inter path.
   */
  bool IsInInterPath(const te::Operation &op);
  /*!
   * \brief Get the spatial extents of first op.
   */
  std::vector<int> GetSpatialExtentsByInferBound(te::Schedule sch,
                                                 const te::Operation &op);
  /*!
   * \brief Get the reduce extents of first op.
   */
  std::vector<int> GetReduceExtentsByInferBound(te::Schedule sch,
                                                const te::Operation &op);
  /*!
   * \brief Get the batch-like dimension index.
   */
  std::vector<int> GetBatchLikeDim(const te::Operation& op);

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.CUDATensorizeContext";
  TVM_DECLARE_BASE_OBJECT_INFO(CUDATensorizeContextNode, Object);
};

class CUDATensorizeContext : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param layer The layer to schedule
   * \param state The tensorize hyper fusion state
   * \param cuda_param The parameters of GPU
   */
  TVM_DLL CUDATensorizeContext(Layer layer, TensorizeHyperFusionState state,
                               hardware::HardwareParam cuda_param);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CUDATensorizeContext, ObjectRef,
                                        CUDATensorizeContextNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CUDATensorizeContextNode);
};

/*!
 * \brief A class for cuda schedule param.
 */
class CUDATensorizeParamNode : public Object {
public:
  /*! \brief The warp size */
  int warp_size;
  /*! \brief The threadIdx.y size */
  int ty_size;
  /*! \brief The threadIdx.z size */
  int tz_size;
  /*! \brief The input vector load length */
  int input_vector_len;
  /*! \brief The serial inner loop for y */
  int serial_y;
  /*! \brief The serial inner loop for z */
  int serial_z;
  /*! \brief The block reduce x dim */
  int block_rx;
  /*! \brief The block reduce y dim */
  int block_ry;
  /*! \brief The block reduce z dim (for future) */
  int block_rz;
  /*! \brief The warp reduce x dim */
  int warp_rx;
  /*! \brief The warp reduce y dim */
  int warp_ry;
  /*! \brief The warp reduce z dim (for future) */
  int warp_rz;
  /*! \brief The unroll steps */
  int unroll_steps;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("warp_size", &warp_size);
    v->Visit("ty_size", &ty_size);
    v->Visit("tz_size", &tz_size);
    v->Visit("input_vector_len", &input_vector_len);
    v->Visit("serial_y", &serial_y);
    v->Visit("serial_z", &serial_z);
    v->Visit("block_rx", &block_rx);
    v->Visit("block_ry", &block_ry);
    v->Visit("block_rz", &block_rz);
    v->Visit("warp_rx", &warp_rx);
    v->Visit("warp_ry", &warp_ry);
    v->Visit("warp_rz", &warp_rz);
    v->Visit("unroll_steps", &unroll_steps);
  }

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.CUDATensorizeParam";
  TVM_DECLARE_BASE_OBJECT_INFO(CUDATensorizeParamNode, Object);
};

class CUDATensorizeParam : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param warp_size The warp size
   * \param ty_size The threadIdx.y size
   * \param tz_size The threadIdx.z size
   * \param input_vector_len The input vector length
   * \param serial_y The y dim serial trip count
   * \param serial_z The z dim serial trip count
   * \param block_rx The block x reduce trip count
   * \param block_ry The block y reduce trip count
   * \param block_rz The block z reduce trip count
   * \param warp_rx The warp x reduce trip count
   * \param warp_ry The warp y reduce trip count
   * \param warp_rz The warp z reduce trip count
   * \param unroll_steps The unroll steps
   */
  TVM_DLL CUDATensorizeParam(int warp_size, int ty_size, int tz_size,
                             int input_vector_len, int serial_y, int serial_z,
                             int block_rx, int block_ry, int block_rz,
                             int warp_rx, int warp_ry, int warp_rz,
                             int unroll_steps);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CUDATensorizeParam, ObjectRef,
                                        CUDATensorizeParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CUDATensorizeParamNode);
};

te::Schedule tensorize_cuda(Layer layer, TensorizeHyperFusionState state,
                            hardware::HardwareParam cuda_param,
                            CUDATensorizeParam tensorize_param);

} // namespace auto_tensorize

} // namespace ditto