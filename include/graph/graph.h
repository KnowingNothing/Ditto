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

namespace graph {

bool IsConstInt(const PrimExpr &expr);

class SubstituteTensor : public tir::ExprMutator {
public:
  using tir::ExprMutator::VisitExpr;

  SubstituteTensor(Array<te::Tensor> org, Array<te::Tensor> replace)
      : org_(org), replace_(replace) {}

private:
  Array<te::Tensor> org_;
  Array<te::Tensor> replace_;

protected:
  using tir::ExprMutator::VisitExpr_;
  // list of functions to override.
  PrimExpr VisitExpr_(const tir::ProducerLoadNode *op) override;
};

class ExistVar : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;

  ExistVar(tir::Var var) : var_(var) {}

  bool operator()(const PrimExpr &expr) {
    VisitExpr(expr);
    return exist_;
  }

private:
  tir::Var var_;
  bool exist_{false};

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::VarNode *op) override;
};

// fwd decl for LayerNode
class LayerNode;
// fwd decl for LayerTensor
class LayerTensor;

/*!
 * \brief Layer class.
 */
class Layer : public ObjectRef {
public:
  // TVM_DEFINE_OBJECT_REF_METHODS(Layer, ObjectRef, LayerNode);
  // TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerNode);

  /*! \brief default constructor  */
  Layer() {}
  explicit Layer(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline LayerNode *operator->() const;
  /*!
   * \brief The constructor.
   * \param name The name of layer
   * \param ops The op of this layer
   * \param inputs The inputs of this layer
   * \param weights The weights of this layer
   * \param const_scalars The constant scalars
   * \param const_tensors The constant tensors
   * \param gradients The gradients of this layer
   */
  TVM_DLL Layer(std::string name, Array<te::Operation> ops,
                Array<te::Tensor> inputs, Array<te::Tensor> weights,
                Array<PrimExpr> const_scalars, Array<te::Tensor> const_tensors,
                Array<te::Tensor> gradients);
  /*!
   * \brief Self-checking if the given compute is valid.
   */
  void CheckValidity();
  /*!
   * \brief The constructor.
   * \param inputs The input tensors.
   */
  std::vector<LayerTensor>
  ProduceOutputs(std::vector<LayerTensor> layer_inputs);
  /*! \brief specify container node */
  using ContainerType = LayerNode;
};

/////////////////////////////////////
// Definitions for tensor between layers
////////////////////////////////////

/*!
 * \brief LayerTensorNode class.
 */
class LayerTensorNode : public Object {
public:
  /*! \brief The name of layer (optional) */
  std::string name{"layer_tensor"};
  /*! \brief The layer that produces this tensor, can be nullptr */
  Layer layer{nullptr};
  /*! \brief The real tensor wrapped */
  te::Tensor tensor;
  /*! \brief The ordinal number of this tensor */
  int value_idx{0};

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("layer", &layer);
    v->Visit("tensor", &tensor);
    v->Visit("value_idx", &value_idx);
  }

  static constexpr const char *_type_key = "ditto.LayerTensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayerTensorNode, Object);
};

class LayerTensor : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of layer
   * \param layer The layer
   * \param tensor The tensor
   * \param value_idx The value index
   */
  TVM_DLL LayerTensor(std::string name, Layer layer, te::Tensor tensor,
                      int value_idx);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LayerTensor, ObjectRef,
                                        LayerTensorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerTensorNode);
};

/////////////////////////////////////
// Definitions for layer
////////////////////////////////////

/*!
 * \brief LayerNode class.
 */
class LayerNode : public Object {
public:
  /*! \brief The name of layer (optional) */
  std::string name{"layer"};
  /*! \brief The output ops within this layer, required */
  Array<te::Operation> ops;
  /*! \brief The inputs of this layer, can by [] */
  Array<te::Tensor> inputs;
  /*! \brief The weights of this layer, can by [] */
  Array<te::Tensor> weights;
  /*! \brief The const scalar values of this layer, can by [] */
  Array<PrimExpr> const_scalars;
  /*! \brief The const tensors of this layer, can by [] */
  Array<te::Tensor> const_tensors;
  /*! \brief The gradients of this layer, can by [] */
  Array<te::Tensor> gradients;
  /*! \brief The input layer tensors */
  std::vector<LayerTensor> input_layer_tensors_;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("ops", &ops);
    v->Visit("inputs", &inputs);
    v->Visit("weights", &weights);
    v->Visit("const_scalars", &const_scalars);
    v->Visit("const_tensors", &const_tensors);
    v->Visit("gradients", &gradients);
  }
  /*!
   * \brief Get the input tensors.
   */
  Array<LayerTensor> InputTensors() const;
  /*!
   * \brief Get all the ops within this layer.
   * from outputs to inputs
   */
  Array<te::Operation> GetAllOps() const;

  static constexpr const char *_type_key = "ditto.Layer";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayerNode, Object);
};

inline LayerNode *Layer::operator->() const {
  return static_cast<LayerNode *>(data_.get());
}

/////////////////////////////////////
// Definitions for Block
// Block has only one input and one output
// The input shape and output shape
// of one block are fixed
////////////////////////////////////

/*!
 * \brief A base class for block.
 */
class BlockNode : public Object {
public:
  /*! \brief The name of block */
  std::string name;
  /*! \brief The output tensors */
  Array<LayerTensor> out_tensors;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("out_tensors", &out_tensors);
  }
  /*!
   * \brief Get all the layers within this block.
   * from outputs to inputs
   */
  Array<Layer> GetAllLayers() const;

  static constexpr const char *_type_key = "ditto.Block";
  TVM_DECLARE_BASE_OBJECT_INFO(BlockNode, Object);
};

class Block : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of block
   */
  TVM_DLL Block(std::string name, Array<LayerTensor> out_tensors);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Block, ObjectRef, BlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockNode);
};

/////////////////////////////////////
// Definitions for Graph
// Graph = List[Block]
////////////////////////////////////

/*!
 * \brief A base class for graph.
 */
class GraphNode : public Object {
public:
  /*! \brief The name of graph */
  std::string name;
  /*! \brief The list of blocks */
  Array<Block> block_list;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("block_list", &block_list);
  }

  static constexpr const char *_type_key = "ditto.Graph";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphNode, Object);
};

class Graph : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   * \param block_list
   */
  TVM_DLL Graph(std::string name, Array<Block> block_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Graph, ObjectRef, GraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GraphNode);
};

} // namespace graph

} // namespace ditto

namespace std {
template <> struct hash<::ditto::graph::Layer> : public ::tvm::ObjectPtrHash {};

template <> struct hash<::ditto::graph::LayerTensor> {
  std::size_t operator()(const ::ditto::graph::LayerTensor &k) const {
    ::tvm::ObjectPtrHash hasher;
    if (k.defined() && k->layer.defined()) {
      return hasher(k->layer);
    } else {
      return hasher(k);
    }
  }
};
} // namespace std

namespace ditto {

namespace graph {
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

  static constexpr const char *_type_key = "ditto.TensorState";
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

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("op", &op); }
  /*!
   * \brief Split the dimension of output tensor.
   * \param iv The dimension
   * \param factor The split factor
   * \param p_outer
   * \param p_inner
   * \param ordinal
   */
  void SplitSpatial(tir::IterVar iv, PrimExpr factor, tir::IterVar *p_outer,
                    tir::IterVar *p_inner, int *ordinal);
  /*!
   * \brief Split the reduce axis.
   * \param iv The dimension
   * \param factor The split factor
   * \param p_outer
   * \param p_inner
   * \param ordinal
   */
  void SplitReduce(tir::IterVar iv, PrimExpr factor, tir::IterVar *p_outer,
                   tir::IterVar *p_inner, int *ordinal);
  /*!
   * \brief Split the axis.
   * \param iv The dimension
   * \param factor The split factor
   * \param p_outer
   * \param p_inner
   * \param ordinal
   */
  void Split(tir::IterVar iv, PrimExpr factor, tir::IterVar *p_outer,
             tir::IterVar *p_inner, int *ordinal);
  /*!
   * \brief Propagate split from producer to consumer.
   * \param op The producer
   * \param ordinal The ordinal number of axis
   * \param outer
   * \param inner
   */
  void PropagateSplit(te::Operation op, int ordinal, tir::IterVar outer,
                      tir::IterVar inner);

  static constexpr const char *_type_key = "ditto.OpState";
  TVM_DECLARE_BASE_OBJECT_INFO(OpStateNode, Object);

private:
  /*!
   * \brief Change the access of one input without global rewrite.
   * \param op The producer that produces the tensor.
   * \param access_idx The array of indices
   */
  void _SetReadAccess(te::Operation op, Array<PrimExpr> access_idx);
  /*!
   * \brief Change the dimension of output with rewrite.
   * \param new_axis The new axis
   * \param new_reduce_axis The new reduce_axis
   * \param mapping The binary relation from original vars to new vars
   */
  void _SetWriteAccess(Array<tir::IterVar> new_axis,
                       Array<tir::IterVar> new_reduce_axis,
                       Map<tir::Var, PrimExpr> mapping);
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
  std::unordered_map<te::Operation, std::unordered_set<te::Operation>>
      read_graph;
  std::unordered_map<te::Operation, std::unordered_set<te::Operation>>
      feed_graph;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("layer", &layer); }

  /*!
   * \brief Split the axis of one op.
   * \param op
   * \param iv The dimension
   * \param factor The split factor
   * \param p_outer
   * \param p_inner
   * \param ordinal
   */
  void Split(te::Operation op, tir::IterVar iv, PrimExpr factor,
             tir::IterVar *p_outer, tir::IterVar *p_inner, int *ordinal);

  static constexpr const char *_type_key = "ditto.LayerState";
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

/*!
 * \brief A base class for block stage.
 */
class BlockStateNode : public Object {
public:
  /*! \brief The block */
  Block block;
  std::unordered_map<Layer, LayerState> layer_states;
  std::vector<Layer> all_layers;
  std::unordered_map<Layer, std::unordered_set<Layer>> feed_graph;

  void VisitAttrs(tvm::AttrVisitor *v) { v->Visit("block", &block); }

  /*!
   * \brief Split the axis of one op.
   * \param layer
   * \param op
   * \param iv The dimension
   * \param factor The split factor
   * \param p_outer
   * \param p_inner
   * \param ordinal
   */
  void Split(Layer layer, te::Operation op, tir::IterVar iv, PrimExpr factor,
             tir::IterVar *p_outer, tir::IterVar *p_inner, int *ordinal);

  static constexpr const char *_type_key = "ditto.BlockState";
  TVM_DECLARE_BASE_OBJECT_INFO(BlockStateNode, Object);
};

class BlockState : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param block The block
   */
  TVM_DLL BlockState(Block block);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BlockState, ObjectRef, BlockStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockStateNode);
};

} // namespace graph

} // namespace ditto