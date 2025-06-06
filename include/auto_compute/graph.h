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

namespace auto_compute {

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
   */
  TVM_DLL Layer(std::string name, Array<te::Operation> ops,
                Array<te::Tensor> inputs, Array<te::Tensor> weights,
                Array<PrimExpr> const_scalars, Array<te::Tensor> const_tensors);
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

  static constexpr const char *_type_key = "ditto.auto_compute.LayerTensor";
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

  inline bool operator==(const LayerTensor &other) const {
    if (get() == other.get())
      return true;
    if (get() == nullptr || other.get() == nullptr)
      return false;
    if ((*this)->layer.defined() || other->layer.defined()) {
      return (*this)->layer == other->layer &&
             (*this)->value_idx == other->value_idx;
    } else {
      return false;
    }
  }

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
  Array<PrimExpr> const_scalars; // this is for future usage, not used now
  /*! \brief The const tensors of this layer, can by [] */
  Array<te::Tensor> const_tensors;
  /*! \brief The input layer tensors */
  std::vector<LayerTensor> input_layer_tensors_;
  /*! \brief The output layer tensors */
  std::vector<LayerTensor> output_layer_tensors_;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("ops", &ops);
    v->Visit("inputs", &inputs);
    v->Visit("weights", &weights);
    v->Visit("const_scalars", &const_scalars);
    v->Visit("const_tensors", &const_tensors);
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
  /*!
   * \brief Get the string fingerprint of this layer.
   */
  std::string GetFingerprint() const;
  /*!
   * \brief Get the total data read/write for this layer.
   */
  FloatImm GetDataTransferAmount() const;
  /*!
   * \brief Get the total GFLOPS for this layer.
   */
  FloatImm GetGFLOPS() const;

  static constexpr const char *_type_key = "ditto.auto_compute.Layer";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayerNode, Object);
};

inline LayerNode *Layer::operator->() const {
  return static_cast<LayerNode *>(data_.get());
}

// /////////////////////////////////////
// // Definitions for Block
// // Block has only one input and one output
// // The input shape and output shape
// // of one block are fixed
// ////////////////////////////////////

// /*!
//  * \brief A base class for block.
//  */
// class BlockNode : public Object {
// public:
//   /*! \brief The name of block */
//   std::string name;
//   /*! \brief The output tensors */
//   Array<LayerTensor> out_tensors;

//   void VisitAttrs(tvm::AttrVisitor *v) {
//     v->Visit("name", &name);
//     v->Visit("out_tensors", &out_tensors);
//   }

//   static constexpr const char *_type_key = "ditto.auto_compute.Block";
//   TVM_DECLARE_BASE_OBJECT_INFO(BlockNode, Object);
// };

// class Block : public ObjectRef {
// public:
//   /*!
//    * \brief The constructor.
//    * \param name The name of block
//    */
//   TVM_DLL Block(std::string name, Array<LayerTensor> out_tensors);

//   TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Block, ObjectRef, BlockNode);
//   TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockNode);
// };

/////////////////////////////////////
// Definitions for Graph
////////////////////////////////////

/*!
 * \brief A base class for graph.
 */
class GraphNode : public Object {
public:
  /*! \brief The name of graph */
  std::string name;
  /*! \brief The list of layer tensors as inputs */
  Array<LayerTensor> graph_inputs;
  /*! \brief The list of layer tensors as outputs */
  Array<LayerTensor> graph_outputs;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("graph_inputs", &graph_inputs);
    v->Visit("graph_outputs", &graph_outputs);
  }

  /*!
   * \brief Get all the layers within this graph.
   * from outputs to inputs
   */
  Array<Layer> GetAllLayers() const;

  static constexpr const char *_type_key = "ditto.auto_compute.Graph";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphNode, Object);
};

/*!
 * \brief Get all the layers within the source and sink.
 */
Array<Layer> FindConvexSet(Array<Layer> source, Array<Layer> sink);

class Graph : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   * \param graph_inputs The inputs for graph
   * \param graph_outputs The outputs for graph
   */
  TVM_DLL Graph(std::string name, Array<LayerTensor> graph_inputs,
                Array<LayerTensor> graph_outputs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Graph, ObjectRef, GraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GraphNode);
};

} // namespace auto_compute

} // namespace ditto

namespace std {
template <>
struct hash<::ditto::auto_compute::Layer> : public ::tvm::ObjectPtrHash {};

template <> struct hash<::ditto::auto_compute::LayerTensor> {
  std::size_t operator()(const ::ditto::auto_compute::LayerTensor &k) const {
    ::tvm::ObjectPtrHash hasher;
    if (k.defined() && k->layer.defined()) {
      return hasher(k->layer);
    } else {
      return hasher(k);
    }
  }
};
} // namespace std
