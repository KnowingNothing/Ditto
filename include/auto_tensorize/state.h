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

/*!
 * \brief A class for access function.
 */
class AccessFunctionNode : public Object {
public:
  /*! \brief The tensor op to access */
  te::Operation op;
  /*! \brief The indices for one tensor */
  Array<Array<PrimExpr>> access_indices;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("op", &op);
    v->Visit("access_indices", &access_indices);
  }

  static constexpr const char *_type_key = "ditto.auto_tensorize.AccessFunction";
  TVM_DECLARE_BASE_OBJECT_INFO(AccessFunctionNode, Object);
};

class AccessFunction : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param op The operation
   */
  TVM_DLL AccessFunction(te::Operation op,
                         Array<Array<PrimExpr>> access_indices);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AccessFunction, ObjectRef,
                                        AccessFunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AccessFunctionNode);
};

/*!
 * \brief A class for op state.
 */
class OpHyperStateNode : public Object {
public:
  /*! \brief The layer state */
  te::Operation op;
  /*! \brief The pattern of the layer */
  OpPattern pattern;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("op", &op);
    v->Visit("pattern", &pattern);
  }
  /*!
   * \brief Get the spatial iterators.
   */
  Array<tir::IterVar> SpatialIters();
  /*!
   * \brief Get the reduce iterators.
   */
  Array<tir::IterVar> ReduceIters();
  /*!
   * \brief Get the read access functions.
   * The order is the same as InputTensors().
   */
  Array<AccessFunction> ReadAccessFunctions();
  /*!
   * \brief Get the write access function.
   * We only assume one output.
   */
  AccessFunction WriteAccessFunctions();

  static constexpr const char *_type_key = "ditto.auto_tensorize.OpHyperState";
  TVM_DECLARE_BASE_OBJECT_INFO(OpHyperStateNode, Object);
};

class OpHyperState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param op The operation
   */
  TVM_DLL OpHyperState(te::Operation op);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(OpHyperState, ObjectRef,
                                        OpHyperStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OpHyperStateNode);
};

/*!
 * \brief A class for fusion of linear topo.
 */
class SerialFusionStateNode : public Object {
public:
  /*! \brief The layer state */
  LayerState layer_state;
  /*! \brief The compute ops */
  Array<te::Operation> ops;
  /*! \brief The states for ops */
  std::unordered_map<te::Operation, OpHyperState> op_hyper_states;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layer_state", &layer_state);
    v->Visit("ops", &ops);
  }

  bool IsLinearTopo();
  int CountOp(OpPattern pattern);

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.SerialFusionState";
  TVM_DECLARE_BASE_OBJECT_INFO(SerialFusionStateNode, Object);
};

class SerialFusionState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param layer The layer
   */
  TVM_DLL SerialFusionState(Layer layer);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SerialFusionState, ObjectRef,
                                        SerialFusionStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SerialFusionStateNode);
};

} // namespace auto_tensorize

} // namespace ditto