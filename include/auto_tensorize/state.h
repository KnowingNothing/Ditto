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
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/pattern.h>

using namespace tvm;
namespace ditto {
using namespace ditto::auto_compute;
namespace auto_tensorize {
/*!
 *  \brief container for SerialFusionState
 */
class SerialFusionStateNode;

/*!
 * \brief A class for op state.
 */
class OpHyperStateNode : public Object {
public:
  /*! \brief The layer state */
  te::Operation op;
  /*! \brief The pattern of the layer */
  OpPattern pattern;
  /*! \brief The index in the layer*/
  size_t index;
  /*! \brief The iterVar map*/
  std::unordered_map<tir::IterVar, IterVar> IterMap;
  /*! \brief The axes for tensorize */
  Array<te::IterVar> tensorizeAxes;
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("op", &op);
    v->Visit("pattern", &pattern);
  }
  /*!
   *  \brief get all the iters.
   *  \return  the IterVar Array.
   */
  Array<IterVar> getAllIters();
  /*!
   *  \brief get all the iters.
   *  \return the map from tir::IterVar to IterVar;
   */
  std::unordered_map<tir::IterVar, IterVar> getIterMap() { return IterMap; }
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
  /*!
   * \brief Get the first computeOp pos in input tensors.
   */
  int getFirstProducerPos() {
    int idx = 0;
    for (auto it : op->InputTensors()) {
      if (it->op.as<te::ComputeOpNode>())
        return idx;
      idx++;
    }
    return -1;
  }
  /*! get the idx to var mapping*/
  std::unordered_map<int, tir::Var> getIdx2var() {
    std::unordered_map<int, tir::Var> ret;
    for (auto iv : getAllIters())
      ret[iv->index] = iv->originVar;
    return ret;
  }
  void registerTensorizeAxes(Array<tir::IterVar> tensorizeAxes_){
    tensorizeAxes = tensorizeAxes_;
  }
  /*! get all vars*/
  Array<tir::Var> getAllVars();

  Map<tir::Var, IntImm> getBounds();

  static constexpr const char *_type_key = "ditto.auto_tensorize.OpHyperState";
  TVM_DECLARE_BASE_OBJECT_INFO(OpHyperStateNode, Object);
};

class OpHyperState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param op The operation
   */
  TVM_DLL OpHyperState(te::Operation op, size_t idx);

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
  /*! \brief the firstOp */
  te::Operation first_op;
  /*! \brief the firstOp */
  te::Operation second_op;
  /*!
   *  \brief The states for ops
   */
  std::unordered_map<te::Operation, OpHyperState> op_hyper_states;
  /*! \brief The tensorize axes */
  Array<tir::IterVar> tensorizeAxes;
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layer_state", &layer_state);
    v->Visit("ops", &ops);
    v->Visit("first_op", &first_op);
    v->Visit("second_op", &second_op);
  }

  bool IsLinearTopo();
  int CountOp(OpPattern pattern);
  /*!
   * \brief get the cubic state pairs
   * \return <op1, op2>, op2 takes op1 as input (may be indirect)
   */
  std::pair<OpHyperState, OpHyperState> getCubicOpPair();
  void registerTensorizeAxes(Array<tir::IterVar> _tensorizeAxes){
    tensorizeAxes = _tensorizeAxes;
    for (auto & k: op_hyper_states)
      k.second->registerTensorizeAxes(tensorizeAxes);
  }
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

/*!
 * \brief The function to validate if a layer is applicable.
 * \param state The fusion state
 */
std::pair<bool, std::string> validate(SerialFusionState state);

/*!
 * \brief The SerialFusionState builder
 * \param layer The layer to schedule
 */
inline SerialFusionState
buildSerialFusionState(Layer layer) {
  SerialFusionState rv = SerialFusionState(layer);
  bool valid = false;
  std::string reason = "";
  std::tie(valid, reason) = validate(rv);
  CHECK(valid) << "The given layer is not suitable for auto tensorize because "
               << reason << "The layer is: " << std::endl
               << layer << "\n";
  return rv;
}

/*!
 * \brief The IterGraph builder
 * \param
 */
IterGraph buildIterGraph(SerialFusionState sfState,
                         Array<te::IterVar> tensorizeAxes,
                          String path);

inline OpHyperState buildOpHyperState(te::Operation op, size_t idx) {
  return OpHyperState(op, idx);
}

} // namespace auto_tensorize

} // namespace ditto