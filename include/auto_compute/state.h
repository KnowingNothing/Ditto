#pragma once

#include <auto_compute/graph.h>
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

namespace ditto {

namespace auto_compute {
/////////////////////////////////////
// Definitions for TensorState, OpState,
// LayerState, and GraphState
////////////////////////////////////

/*!
 * \brief A base class for tensor state.
 */
class TensorStateNode : public Object {
public:
  /*! \brief The tensor */
  te::Tensor tensor;
  std::vector<PrimExpr> shape;
  std::vector<PrimExpr> access_index;
  runtime::DataType dtype;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("tensor", &tensor); }
  /*!
   * \brief Return the dimension of tensor.
   */
  inline int nDim() const { return (int)(this->access_index.size()); }
  /*!
   * \brief Split the dimension of one tensor.
   * \param ordinal The dimension ordinal number
   * \param outer The outer result
   * \param inner The inner result
   */
  void SplitDim(int ordinal, tir::IterVar outer, tir::IterVar inner);
  /*!
   * \brief Substitute one var with an expression.
   * \param v The var
   * \param expr The expression
   */
  void SubstituteIndexVar(tir::Var v, PrimExpr expr);
  /*!
   * \brief Substitute vars through a map.
   * \param mapping The mapping
   */
  void SubstituteIndexVars(Map<tir::Var, PrimExpr> mapping);

  static constexpr const char *_type_key = "ditto.auto_compute.TensorState";
  TVM_DECLARE_BASE_OBJECT_INFO(TensorStateNode, Object);
};

class TensorState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param tensor The tensor
   */
  TVM_DLL TensorState(te::Tensor tensor, Array<PrimExpr> access_index);
  /*!
   * \brief Returns if an variable used in access index.
   * \param v The var
   */
  std::pair<bool, int> ContainIndex(tir::Var v) const;
  /*!
   * \brief Returns if the index is direct access.
   * \param index The index
   */
  static bool IsSimpleIndex(PrimExpr index) {
    return (index.as<tir::VarNode>() != nullptr);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorState, ObjectRef,
                                        TensorStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorStateNode);
};

/*!
 * \brief A base class for opstage.
 */
class OpStateNode : public Object {
public:
  /*! \brief The op */
  te::Operation op;
  std::vector<tir::IterVar> axis;
  std::vector<tir::IterVar> reduce_axis;
  runtime::DataType dtype;
  std::unordered_map<te::Operation, TensorState> input_tensor_states;
  std::unordered_map<te::Operation, te::Operation> op_mapping;

  class BodyVisitor : public tir::ExprVisitor {
  public:
    using tir::ExprVisitor::VisitExpr;

    BodyVisitor(runtime::ObjectPtr<OpStateNode> self) : self_(self) {}

  protected:
    using tir::ExprVisitor::VisitExpr_;
    void VisitExpr_(const tir::ProducerLoadNode *op) override;

  private:
    runtime::ObjectPtr<OpStateNode> self_;
  };

  class ReplaceInputs : public tir::ExprMutator {
  public:
    using tir::ExprMutator::VisitExpr;

    ReplaceInputs(OpStateNode *self, Map<te::Operation, te::Operation> mapping)
        : self_(self), mapping_(mapping) {}

  protected:
    using tir::ExprMutator::VisitExpr_;
    PrimExpr VisitExpr_(const tir::ProducerLoadNode *op) override;

  private:
    OpStateNode *self_;
    Map<te::Operation, te::Operation> mapping_;
  };

  class RemapInput : public tir::ExprMutator {
  public:
    using tir::ExprMutator::VisitExpr;

    RemapInput(OpStateNode *self, te::Operation original_producer,
               te::Operation new_producer, Array<tir::Var> original_vars,
               Array<tir::Var> new_vars,
               Map<tir::Var, PrimExpr> new_vars_mapping)
        : self_(self), original_producer_(original_producer),
          new_producer_(new_producer), original_vars_(original_vars),
          new_vars_(new_vars), new_vars_mapping_(new_vars_mapping) {}

  protected:
    using tir::ExprMutator::VisitExpr_;
    PrimExpr VisitExpr_(const tir::ProducerLoadNode *op) override;

  private:
    OpStateNode *self_;
    te::Operation original_producer_, new_producer_;
    Array<tir::Var> original_vars_;
    Array<tir::Var> new_vars_;
    Map<tir::Var, PrimExpr> new_vars_mapping_;
  };

  /*!
   * \brief Get the axis.
   */
  Array<tir::IterVar> GetAxis() const;

  /*!
   * \brief Get the reduce axis.
   */
  Array<tir::IterVar> GetReduceAxis() const;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("op", &op); }

  /*!
   * \brief Transform one op and isolate the correction op.
   * \param spatial_vars
   * \param spatial_forward Affine expressions of spatial vars
   * \param spatial_backward Reverse spatial transformation
   * \param reduce_vars
   * \param reduce_forward Affine expressions of reduce vars
   * \param reduce_backward Reverse reduce transformation
   */
  std::pair<te::Operation, te::Operation> TransformIsolation(
      Array<tir::Var> spatial_vars, Array<PrimExpr> spatial_forward,
      Array<PrimExpr> spatial_backward, Array<tir::Var> reduce_vars,
      Array<PrimExpr> reduce_forward, Array<PrimExpr> reduce_backward);

  /*!
   * \brief Transform input access.
   * \param original_producer_location The first place producer occurs
   * \param new_producer New producer op
   * \param original_vars Original producer spatial vars
   * \param new_vars New producer spatial vars
   * \param new_vars_mapping Mapping from new vars to original vars
   */
  te::Operation TransformRemapInput(int original_producer_location,
                                    te::Operation new_producer,
                                    Array<tir::Var> original_vars,
                                    Array<tir::Var> new_vars,
                                    Map<tir::Var, PrimExpr> new_vars_mapping);

  /*!
   * \brief Make the transformed compute.
   * \param inputs The input tensors
   */
  te::Operation MakeCompute(Array<te::Tensor> inputs);

  static constexpr const char *_type_key = "ditto.auto_compute.OpState";
  TVM_DECLARE_BASE_OBJECT_INFO(OpStateNode, Object);
};

class OpState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param op The op
   */
  TVM_DLL OpState(te::Operation op);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(OpState, ObjectRef, OpStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OpStateNode);
};

/*!
 * \brief A base class for layer stage.
 */
class LayerStateNode : public Object {
public:
  /*! \brief The layer */
  Layer layer;
  std::unordered_map<te::Operation, OpState> op_states;
  std::vector<te::Operation> all_ops;
  std::unordered_map<te::Operation, std::vector<te::Operation>> read_graph;
  std::unordered_map<te::Operation, std::unordered_set<te::Operation>>
      feed_graph;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("layer", &layer); }

  /*!
   * \brief Get op state.
   * \param op
   */
  OpState GetOpState(te::Operation op) const;

  /*!
   * \brief Transform write access.
   * \param op The op to be transformed
   * \param spatial_vars
   * \param spatial_forward from old vars to new vars
   * \param spatial_backward from new vars to old vars
   * \param reduce_vars
   * \param reduce_forward from old vars to new vars
   * \param reduce_backward from new vars to old vars
   * \param explicit_transform
   */
  te::Operation
  Transform(te::Operation op, Array<tir::Var> spatial_vars,
            Array<PrimExpr> spatial_forward, Array<PrimExpr> spatial_backward,
            Array<tir::Var> reduce_vars, Array<PrimExpr> reduce_forward,
            Array<PrimExpr> reduce_backward, bool explicit_transform = true);

  /*!
   * \brief Make the transformed compute.
   * \param inputs The input tensors
   */
  Layer MakeCompute(Array<LayerTensor> inputs);
  /*!
   * \brief Get the current ops in this layer state.
   */
  Array<te::Operation> GetCurrentOps();

  static constexpr const char *_type_key = "ditto.auto_compute.LayerState";
  TVM_DECLARE_BASE_OBJECT_INFO(LayerStateNode, Object);
};

class LayerState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param layer The layer
   */
  TVM_DLL LayerState(Layer layer);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LayerState, ObjectRef, LayerStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerStateNode);
};

// /*!
//  * \brief A base class for block stage.
//  */
// class BlockStateNode : public Object {
// public:
//   /*! \brief The block */
//   Block block;
//   std::unordered_map<Layer, LayerState> layer_states;
//   std::vector<Layer> all_layers;
//   std::unordered_map<Layer, std::vector<Layer>> read_graph;
//   std::unordered_map<Layer, std::unordered_set<Layer>> feed_graph;

//   void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("block", &block); }

//   static constexpr const char *_type_key = "ditto.auto_compute.BlockState";
//   TVM_DECLARE_BASE_OBJECT_INFO(BlockStateNode, Object);
// };

// class BlockState : public ObjectRef {
// public:
//   /*!
//    * \brief The constructor.
//    * \param block The block
//    */
//   TVM_DLL BlockState(Block block);

//   TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BlockState, ObjectRef,
//   BlockStateNode); TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockStateNode);
// };

/*!
 * \brief A base class for graph stage.
 */
class GraphStateNode : public Object {
public:
  /*! \brief The graph */
  Graph graph;
  std::vector<LayerTensor> inputs;
  std::vector<LayerTensor> outputs;
  std::unordered_map<Layer, LayerState> layer_states;
  std::vector<Layer> all_layers;
  std::unordered_map<Layer, std::vector<LayerTensor>> consume_graph;
  std::unordered_map<Layer, std::vector<LayerTensor>> produce_graph;
  std::unordered_map<LayerTensor, std::vector<std::pair<Layer, int>>> feed_graph;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("graph", &graph); }

  /*!
   * \brief Get layer state.
   * \param layer
   */
  LayerState GetLayerState(Layer layer) const;

  /*!
   * \brief Make the transformed compute.
   * \param inputs The input tensors
   */
  Graph MakeCompute(Array<LayerTensor> inputs);
  /*!
   * \brief Get the current ops in this layer state.
   */
  Array<Layer> GetCurrentLayers();
  /*!
   * \brief Partition a layer to fine-grained sub-layers.
   */
  Array<Layer> NormalizePartition(Layer layer, bool modify=true);
  /*!
   * \brief Fuse the convex set between front and back.
   */
  Layer Fuse(Layer front, Layer back, bool modify=true);

  static constexpr const char *_type_key = "ditto.auto_compute.GraphState";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphStateNode, Object);
};

class GraphState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param graph The graph
   */
  TVM_DLL GraphState(Graph graph);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GraphState, ObjectRef, GraphStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GraphStateNode);
};

} // namespace auto_compute

} // namespace ditto