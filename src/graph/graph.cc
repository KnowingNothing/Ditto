#include <graph/graph.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/op.h>
#include <utils/iter_domain.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

using namespace tvm;
namespace ditto {

namespace graph {

TVM_REGISTER_NODE_TYPE(LayerTensorNode);
TVM_REGISTER_NODE_TYPE(LayerNode);
TVM_REGISTER_NODE_TYPE(BlockNode);
TVM_REGISTER_NODE_TYPE(GraphNode);
TVM_REGISTER_NODE_TYPE(TensorStateNode);
TVM_REGISTER_NODE_TYPE(OpStateNode);
TVM_REGISTER_NODE_TYPE(LayerStateNode);
TVM_REGISTER_NODE_TYPE(BlockStateNode);

bool IsConstInt(const PrimExpr &expr) {
  return (expr.as<tir::IntImmNode>() != nullptr);
}

PrimExpr SubstituteTensor::VisitExpr_(const tir::ProducerLoadNode *op) {
  int i = 0;
  for (auto t : org_) {
    if (t == runtime::Downcast<te::Tensor>(op->producer)) {
      return tir::ProducerLoad(replace_[i], op->indices);
    }
    i += 1;
  }
  return tir::ExprMutator::VisitExpr_(op);
}

void ExistVar::VisitExpr_(const tir::VarNode *op) {
  if (exist_)
    return;
  if (op == var_.get()) {
    exist_ = true;
  }
  return;
}

LayerTensor::LayerTensor(std::string name, Layer layer, te::Tensor tensor,
                         int value_idx) {
  auto node = make_object<LayerTensorNode>();
  node->name = name;
  node->layer = layer;
  node->tensor = tensor;
  node->value_idx = value_idx;
  data_ = node;
}

Array<LayerTensor> LayerNode::InputTensors() const {
  CHECK(input_layer_tensors_.size()) << "The input layer tensors for layer "
                                     << _type_key << " has not been set.\n";
  return Array<LayerTensor>(input_layer_tensors_);
}

Array<te::Operation> LayerNode::GetAllOps() const {
  std::unordered_set<te::Operation> visit;
  Array<te::Operation> ret;

  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (visit.count(op))
      return;
    visit.insert(op);
    ret.push_back(op);
    for (auto inp : op->InputTensors()) {
      helper(inp->op);
    }
  };

  for (auto op : ops) {
    helper(op);
  }

  return ret;
}

Layer::Layer(std::string name, Array<te::Operation> ops,
             Array<te::Tensor> inputs, Array<te::Tensor> weights,
             Array<PrimExpr> const_scalars, Array<te::Tensor> const_tensors,
             Array<te::Tensor> gradients) {
  auto node = make_object<LayerNode>();
  node->name = name;
  node->ops = ops;
  node->inputs = inputs;
  node->weights = weights;
  node->const_scalars = const_scalars;
  node->const_tensors = const_tensors;
  node->gradients = gradients;
  data_ = node;
  CheckValidity();
}

void Layer::CheckValidity() {
  /////////////////////////////////////////
  // Check 2 points:
  // 1. each compute node has one output
  // 2. each op is either compute op or placeholder op
  /////////////////////////////////////////

  std::function<void(te::Operation op)> helper;
  std::unordered_set<te::Operation> visit;

  helper = [&](te::Operation op) {
    if (visit.count(op))
      return;
    visit.insert(op);
    for (te::Tensor inp : op->InputTensors()) {
      helper(inp->op);
    }

    CHECK(op->num_outputs() == 1)
        << "Currently only support op with one output.\n";

    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
    CHECK(cop || pop)
        << "Currently only support ComputeOp and PlaceholderOp.\n";
  };

  for (auto op : (*this)->ops) {
    helper(op);
  }
}

std::vector<LayerTensor>
Layer::ProduceOutputs(std::vector<LayerTensor> layer_inputs) {
  auto self = (*this);
  Array<te::Tensor> inputs;
  self->input_layer_tensors_.clear();
  for (auto inp : layer_inputs) {
    inputs.push_back(inp->tensor);
    self->input_layer_tensors_.push_back(inp);
  }
  int num_inputs = (int)(self->inputs.size());
  CHECK((int)inputs.size() == num_inputs)
      << "Expect " << num_inputs << " input tensor but get " << inputs.size()
      << ".\n";
  ///////////////////////////////////////
  // the core context
  std::unordered_map<te::Operation, te::Operation> new_ops;

  for (int i = 0; i < num_inputs; ++i) {
    new_ops[self->inputs[i]->op] = inputs[i]->op;
  }

  ///////////////////////////////////////
  // traversal helper
  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (new_ops.count(op)) {
      return;
    }

    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
    CHECK(cop || pop)
        << "Currently only support ComputeOp and PlaceholderOp.\n";

    Array<te::Tensor> new_tensors;
    for (auto inp : op->InputTensors()) {
      helper(inp->op);
      CHECK(new_ops.count(inp->op)) << "Missing op " << inp->op << ".\n";
      new_tensors.push_back(new_ops.at(inp->op).output(0));
    }

    if (cop) {
      Array<tir::IterVar> new_axis;
      for (tir::IterVar iv : cop->axis) {
        new_axis.push_back(
            tir::IterVar(iv->dom, tir::Var(iv->var->name_hint, iv->var->dtype),
                         iv->iter_type, iv->thread_tag));
      }
      Array<PrimExpr> new_body;
      for (PrimExpr b : cop->body) {
        SubstituteTensor suber(op->InputTensors(), new_tensors);
        new_body.push_back(suber(b));
      }
      te::Operation new_op =
          te::ComputeOp(cop->name, cop->tag, cop->attrs, new_axis, new_body);
      new_ops[op] = new_op;
    } else if (pop) {
      new_ops[op] = op;
    }
  };

  std::vector<LayerTensor> rets;
  int num_out = 0;
  for (te::Operation op : self->ops) {
    helper(op);
    CHECK(new_ops.count(op)) << "Missing op " << op << ".\n";
    rets.push_back(
        LayerTensor(self->name, self, new_ops.at(op).output(0), num_out++));
  }

  return rets;
}

Array<Layer> BlockNode::GetAllLayers() const {
  std::unordered_set<Layer> visit;
  Array<Layer> ret;

  std::function<void(Layer layer)> helper;
  helper = [&](Layer layer) {
    if (visit.count(layer))
      return;
    visit.insert(layer);
    ret.push_back(layer);
    for (auto inp : layer->InputTensors()) {
      helper(inp->layer);
    }
  };

  for (auto t : out_tensors) {
    helper(t->layer);
  }

  return ret;
}

Block::Block(std::string name, Array<LayerTensor> out_tensors) {
  auto node = make_object<BlockNode>();
  node->name = name;
  node->out_tensors = out_tensors;
  data_ = node;
}

Graph::Graph(std::string name, Array<Block> block_list) {
  auto node = make_object<GraphNode>();
  node->name = name;
  node->block_list = block_list;
  data_ = node;
}

void TensorStateNode::SplitDim(int ordinal, tir::IterVar outer,
                               tir::IterVar inner) {
  auto self = this;
  CHECK(0 <= ordinal && ordinal < self->nDim());
  const tir::IntImmNode *const_outer = outer->dom->extent.as<tir::IntImmNode>();
  const tir::IntImmNode *const_inner = inner->dom->extent.as<tir::IntImmNode>();
  CHECK(const_outer && const_inner) << "Expected constant values.\n";
  int outer_extent = const_outer->value;
  int inner_extent = const_inner->value;

  self->shape.erase(self->shape.begin() + ordinal);
  self->shape.insert(self->shape.begin() + ordinal, inner_extent);
  self->shape.insert(self->shape.begin() + ordinal, outer_extent);

  PrimExpr org_index = self->access_index[ordinal];
  PrimExpr outer_index = floordiv(org_index, inner_extent);
  PrimExpr inner_index = floormod(org_index, inner_extent);
  self->access_index.erase(self->access_index.begin() + ordinal);
  self->access_index.insert(self->access_index.begin() + ordinal, inner_index);
  self->access_index.insert(self->access_index.begin() + ordinal, outer_index);
}

void TensorStateNode::SubstituteIndexVar(tir::Var v, PrimExpr expr) {
  int dim = nDim();
  Map<tir::Var, PrimExpr> tmap;
  tmap.Set(v, expr);
  for (int i = 0; i < dim; ++i) {
    this->access_index[i] = tir::Substitute(this->access_index[i], tmap);
  }
}

void TensorStateNode::SubstituteIndexVars(Map<tir::Var, PrimExpr> mapping) {
  int dim = nDim();
  for (int i = 0; i < dim; ++i) {
    this->access_index[i] = tir::Substitute(this->access_index[i], mapping);
  }
}

TensorState::TensorState(te::Tensor tensor, Array<PrimExpr> access_index) {
  auto node = make_object<TensorStateNode>();
  node->tensor = tensor;
  for (auto s : tensor->shape) {
    node->shape.push_back(s);
  }
  for (auto index : access_index) {
    node->access_index.push_back(index);
  }
  node->dtype = tensor->dtype;
  data_ = node;
}

std::pair<bool, int> TensorState::ContainIndex(tir::Var v) const {
  auto self = (*this);
  ExistVar exist(v);
  int which_dim = 0;
  for (PrimExpr expr : self->access_index) {
    if (exist(expr)) {
      return std::make_pair(true, which_dim);
    }
    which_dim += 1;
  }
  return std::make_pair(false, -1);
}

void OpStateNode::BodyVisitor::VisitExpr_(const tir::ProducerLoadNode *op) {
  auto t = runtime::Downcast<te::Tensor>(op->producer);
  if (t.defined() && !self_->input_tensor_states.count(t->op)) {
    self_->input_tensor_states[t->op] = TensorState(t, op->indices);
  }
}

PrimExpr OpStateNode::BodyMutator::VisitExpr_(const tir::ProducerLoadNode *op) {
  auto t = runtime::Downcast<te::Tensor>(op->producer);
  if (t.defined() && mapping_.count(t->op) &&
      self_->input_tensor_states.count(t->op)) {
    te::Operation new_op = mapping_.at(t->op);
    TensorState state = self_->input_tensor_states.at(t->op);
    Array<PrimExpr> new_indices;
    for (auto idx : state->access_index) {
      new_indices.push_back(VisitExpr(idx));
    }
    return tir::ProducerLoad(new_op.output(t->value_index), new_indices);
  }
  Array<PrimExpr> new_indices;
  for (auto idx : op->indices) {
    new_indices.push_back(VisitExpr(idx));
  }
  return tir::ProducerLoad(op->producer, new_indices, op->span);
}

Array<tir::IterVar> OpStateNode::GetAxis() const {
  return Array<tir::IterVar>(axis);
}

Array<tir::IterVar> OpStateNode::GetReduceAxis() const {
  return Array<tir::IterVar>(reduce_axis);
}

void OpStateNode::SplitSpatial(tir::IterVar iv, PrimExpr factor,
                               tir::IterVar *p_outer, tir::IterVar *p_inner,
                               int *ordinal) {
  auto self = this;
  // check if the required axis exists
  bool exist{false};
  int dim{-1};
  for (auto v : self->axis) {
    dim += 1;
    if (v.same_as(iv)) {
      exist = true;
      break;
    }
  }

  CHECK(exist && (dim >= 0)) << "The required axis to split " << iv
                             << " not exists in op " << self->op << ".\n";
  if (ordinal != nullptr)
    *ordinal = dim;

  PrimExpr extent = self->axis[dim]->dom->extent;
  PrimExpr nparts = floordiv(extent + factor - 1, factor);
  arith::Analyzer ana;
  PrimExpr const_nparts = ana.Simplify(nparts);
  PrimExpr const_factor = ana.Simplify(factor);

  ///////////////////////////////////////////
  // It is annoying to handle non-perfect division
  // currently, dynamic bound is ignored here
  // but non-perfect division is not well handled
  // (say we split 17 by factor 4, the results are
  // outer = 5, inner = 4, 5x4=20>17)
  // the good news is that even if we produce a larger
  // output within a block, we can still maintain
  // the correctness of the block.
  // The reason is that the consumer will always
  // iterate within the original bounds.
  CHECK(IsConstInt(const_nparts) && IsConstInt(const_factor))
      << "Currently only support constant extents.\n";

  tir::IterVar outer(Range(tir::make_const(nparts->dtype, 0), const_nparts),
                     tir::Var(iv->var->name_hint + ".o", iv->var->dtype),
                     iv->iter_type, iv->thread_tag);
  tir::IterVar inner(Range(tir::make_const(nparts->dtype, 0), const_factor),
                     tir::Var(iv->var->name_hint + ".i", iv->var->dtype),
                     iv->iter_type, iv->thread_tag);
  if (p_outer != nullptr)
    *p_outer = outer;
  if (p_inner != nullptr)
    *p_inner = inner;

  ////////////////////////////////
  // modify the tensor state
  // for output
  self->axis.erase(self->axis.begin() + dim);
  self->axis.insert(self->axis.begin() + dim, inner);
  self->axis.insert(self->axis.begin() + dim, outer);

  ///////////////////////////////
  // modify the tensor states
  // for inputs, only indices
  PrimExpr new_index = outer->var * factor + inner->var;
  for (auto kv : self->input_tensor_states) {
    kv.second->SubstituteIndexVar(iv->var, new_index);
  }
  /////////////////////////////
  // modify the tensor state
  // of consumer layers is handled
  // by the split function at layer level
  return;
}

void OpStateNode::SplitReduce(tir::IterVar iv, PrimExpr factor,
                              tir::IterVar *p_outer, tir::IterVar *p_inner,
                              int *ordinal) {
  auto self = this;
  // check if the required axis exists
  bool exist{false};
  int dim{-1};
  for (auto v : self->reduce_axis) {
    dim += 1;
    if (v.same_as(iv)) {
      exist = true;
      break;
    }
  }

  CHECK(exist && (dim >= 0)) << "The required axis to split " << iv
                             << " not exists in op " << self->op << ".\n";
  if (ordinal != nullptr)
    *ordinal = dim;

  PrimExpr extent = self->reduce_axis[dim]->dom->extent;
  PrimExpr nparts = floordiv(extent + factor - 1, factor);
  arith::Analyzer ana;
  PrimExpr const_nparts = ana.Simplify(nparts);
  PrimExpr const_factor = ana.Simplify(factor);

  ///////////////////////////////////////////
  // It is annoying to handle non-perfect division
  // currently, dynamic bound is ignored here
  // but non-perfect division is not well handled
  // (say we split 17 by factor 4, the results are
  // outer = 5, inner = 4, 5x4=20>17)
  // the good news is that even if we produce a larger
  // output within a block, we can still maintain
  // the correctness of the block.
  // The reason is that the consumer will always
  // iterate within the original bounds.
  CHECK(IsConstInt(const_nparts) && IsConstInt(const_factor))
      << "Currently only support constant extents.\n";

  tir::IterVar outer(Range(tir::make_const(nparts->dtype, 0), const_nparts),
                     tir::Var(iv->var->name_hint + ".o", iv->var->dtype),
                     iv->iter_type, iv->thread_tag);
  tir::IterVar inner(Range(tir::make_const(nparts->dtype, 0), const_factor),
                     tir::Var(iv->var->name_hint + ".i", iv->var->dtype),
                     iv->iter_type, iv->thread_tag);
  if (p_outer != nullptr)
    *p_outer = outer;
  if (p_inner != nullptr)
    *p_inner = inner;

  ////////////////////////////////
  // modify the tensor state
  // for output
  self->axis.erase(self->reduce_axis.begin() + dim);
  self->axis.insert(self->reduce_axis.begin() + dim, inner);
  self->axis.insert(self->reduce_axis.begin() + dim, outer);

  ///////////////////////////////
  // modify the tensor states
  // for inputs, only indices
  PrimExpr new_index = outer->var * factor + inner->var;
  for (auto kv : self->input_tensor_states) {
    kv.second->SubstituteIndexVar(iv->var, new_index);
  }
  /////////////////////////////
  // modify the tensor state
  // of consumer layers is handled
  // by the split function at layer level
  return;
}

void OpStateNode::Split(tir::IterVar iv, PrimExpr factor, tir::IterVar *p_outer,
                        tir::IterVar *p_inner, int *ordinal) {
  CHECK(iv->iter_type == tir::IterVarType::kCommReduce ||
        iv->iter_type == tir::IterVarType::kDataPar);
  if (iv->iter_type == tir::IterVarType::kCommReduce) {
    SplitReduce(iv, factor, p_outer, p_inner, ordinal);
  } else if (iv->iter_type == tir::IterVarType::kDataPar) {
    SplitSpatial(iv, factor, p_outer, p_inner, ordinal);
  }
}

void OpStateNode::PropagateSplit(te::Operation op, int ordinal,
                                 tir::IterVar outer, tir::IterVar inner) {
  CHECK(input_tensor_states.count(op));
  input_tensor_states.at(op)->SplitDim(ordinal, outer, inner);
}

std::pair<te::Operation, te::Operation> OpStateNode::Transform(
    Array<tir::Var> spatial_vars, Array<PrimExpr> spatial_forward,
    Array<PrimExpr> spatial_backward, Array<tir::Var> reduce_vars,
    Array<PrimExpr> reduce_forward, Array<PrimExpr> reduce_backward) {
  // helper function
  std::function<Array<tir::IterVar>(
      Array<tir::Var> new_vars, Array<PrimExpr> forward,
      Array<PrimExpr> backward, std::vector<tir::IterVar> vars,
      tir::IterVarType type)>
      helper;

  helper = [&](Array<tir::Var> new_vars, Array<PrimExpr> forward,
               Array<PrimExpr> backward, std::vector<tir::IterVar> vars,
               tir::IterVarType type) {
    Map<tir::Var, PrimExpr> var_mapping;
    int count_vars = 0;
    for (auto expr : forward) {
      var_mapping.Set(new_vars[count_vars++], expr);
    }

    Map<tir::Var, Range> original_ranges;
    std::unordered_map<const tir::VarNode *, tir::IterVarType> original_types;
    for (auto iv : vars) {
      original_ranges.Set(iv->var, iv->dom);
      original_types[iv->var.get()] = iv->iter_type;
    }

    Map<tir::Var, Range> new_var_mapping =
        utils::InferRange(var_mapping, original_ranges);
    std::unordered_map<const tir::VarNode *, tir::IterVarType> new_types =
        utils::InferIterVarType(var_mapping, original_types);

    Array<tir::IterVar> new_itervars;
    for (auto var : new_vars) {
      CHECK(new_var_mapping.count(var))
          << "Can't find the var " << var << " after infer range.\n";
      CHECK(new_types.count(var.get()))
          << "Can't find the var " << var << " after infer type.\n";
      CHECK(new_types.at(var.get()) == type)
          << "Expect type " << type << " but get " << new_types.at(var.get())
          << ".\n";
      tir::IterVar new_iter(new_var_mapping.at(var), var, type, "");
      new_itervars.push_back(new_iter);
    }

    return new_itervars;
  };

  // prepare reverse mapping
  Map<tir::Var, PrimExpr> reverse_mapping;

  int num_spatial = (int)(axis.size());
  CHECK(num_spatial == (int)spatial_backward.size())
      << "Spatial length mismatch.\n";
  for (int i = 0; i < num_spatial; ++i) {
    reverse_mapping.Set(axis[i]->var, spatial_backward[i]);
  }

  int num_reduce = (int)(reduce_axis.size());
  CHECK(num_reduce == (int)reduce_backward.size())
      << "Reduce length mismatch.\n";
  for (int i = 0; i < num_reduce; ++i) {
    reverse_mapping.Set(reduce_axis[i]->var, reduce_backward[i]);
  }

  // prepare the forward mapping
  Map<tir::Var, PrimExpr> forward_mapping;
  num_spatial = (int)(spatial_vars.size());
  CHECK(num_spatial == (int)spatial_forward.size())
      << "Spatial length mismatch.\n";
  for (int i = 0; i < num_spatial; ++i) {
    forward_mapping.Set(spatial_vars[i], spatial_forward[i]);
  }

  num_reduce = (int)(reduce_vars.size());
  CHECK(num_reduce == (int)reduce_forward.size())
      << "Reduce length mismatch.\n";
  for (int i = 0; i < num_reduce; ++i) {
    forward_mapping.Set(reduce_vars[i], reduce_forward[i]);
  }

  // prepare the spatial iter vars
  Array<tir::IterVar> upper_axis =
      helper(spatial_vars, spatial_forward, spatial_backward, axis,
             tir::IterVarType::kDataPar);
  // prepare the reduce iter vars
  Array<tir::IterVar> upper_reduce_axis =
      helper(reduce_vars, reduce_forward, reduce_backward, reduce_axis,
             tir::IterVarType::kCommReduce);

  // prepare the new bodies
  const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(pop || cop) << "Expect PlaceholderOp or ComputeOp.\n";
  if (pop) {

    // produce the upper placeholder
    Array<PrimExpr> upper_shape;
    for (auto iv : upper_axis) {
      upper_shape.push_back(iv->dom->extent);
    }

    te::Operation upper =
        te::PlaceholderOp(pop->name + ".upper", upper_shape, pop->dtype);
    // produce the lower compute
    Array<PrimExpr> lower_body;
    Array<PrimExpr> load_index;
    for (auto s : spatial_vars) {
      load_index.push_back(s);
    }

    lower_body.push_back(tir::Substitute(
        tir::ProducerLoad(upper.output(0), load_index), forward_mapping));

    te::Operation lower =
        te::ComputeOp(pop->name + ".lower", pop->name + ".lower", {},
                      Array<tir::IterVar>(axis), lower_body);

    return std::make_pair(upper, lower);
  } else {
    // cop
    // produce the upper compute
    Array<PrimExpr> new_body;
    for (auto b : cop->body) {
      new_body.push_back(tir::Substitute(b, reverse_mapping));
    }
    te::Operation upper =
        te::ComputeOp(cop->name + ".upper", cop->tag + ".upper", cop->attrs,
                      upper_axis, new_body);
    // produce the lower compute
    Array<PrimExpr> lower_body;
    Array<PrimExpr> load_index;
    for (auto s : spatial_vars) {
      load_index.push_back(s);
    }
    lower_body.push_back(tir::Substitute(
        tir::ProducerLoad(upper.output(0), load_index), forward_mapping));
    te::Operation lower =
        te::ComputeOp(cop->name + ".lower", cop->tag + ".lower", cop->attrs,
                      Array<tir::IterVar>(axis), lower_body);
    return std::make_pair(upper, lower);
  }
}

te::Operation OpStateNode::MakeCompute(Array<te::Tensor> inputs) {
  // Warning: the order of inputs is of critical importance
  Array<te::Tensor> original_tensors = op->InputTensors();
  CHECK(original_tensors.size() == inputs.size()) << "Input number mismatch.\n";
  int num_inputs = (int)(original_tensors.size());
  Map<te::Operation, te::Operation> mapping;
  for (int i = 0; i < num_inputs; ++i) {
    mapping.Set(original_tensors[i]->op, inputs[i]->op);
  }
  BodyMutator mutator(this, mapping);
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
  CHECK(cop || pop) << "Only expect PlaceholderOp or ComputeOp.\n";
  if (cop) {
    // compute op
    Array<PrimExpr> new_body;
    for (auto b : cop->body) {
      new_body.push_back(mutator(b));
    }
    return te::ComputeOp(cop->name, cop->tag, cop->attrs, axis, new_body);
  } else {
    // placeholder op
    Array<PrimExpr> shape;
    for (auto iv : axis) {
      shape.push_back(iv->dom->extent);
    }
    return te::PlaceholderOp(pop->name, shape, dtype);
  }
}

void OpStateNode::_SetReadAccess(te::Operation op, Array<PrimExpr> access_idx) {
  // This function is dangeous as it has no checking
  // and may result in wrong compute without carefully selected parameters
  CHECK(input_tensor_states.count(op))
      << "Can't find the op in inputs " << op << ".\n";
  TensorState state = input_tensor_states.at(op);
  state->access_index.clear();
  for (auto idx : access_idx) {
    state->access_index.push_back(idx);
  }
}

void OpStateNode::_SetWriteAccess(Array<tir::IterVar> new_axis,
                                  Array<tir::IterVar> new_reduce_axis,
                                  Map<tir::Var, PrimExpr> mapping) {
  this->axis.clear();
  for (auto iv : new_axis) {
    this->axis.push_back(iv);
  }
  this->reduce_axis.clear();
  for (auto iv : new_reduce_axis) {
    this->reduce_axis.push_back(iv);
  }
  for (auto kv : this->input_tensor_states) {
    kv.second->SubstituteIndexVars(mapping);
  }
}

OpState::OpState(te::Operation op) {
  auto node = make_object<OpStateNode>();
  node->op = op;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
  CHECK(cop || pop) << "Expect PlaceholderOp or ComputeOp.\n";
  if (cop) {
    OpStateNode::BodyVisitor visitor(node);
    for (auto b : cop->body) {
      visitor(b);
    }
    for (auto iv : cop->axis) {
      node->axis.push_back(iv);
    }
    for (auto iv : cop->reduce_axis) {
      node->reduce_axis.push_back(iv);
    }
  } else {
    // placeholder op
    int count_v = 0;
    for (auto s : pop->shape) {
      node->axis.push_back(tir::IterVar(
          Range(0, s),
          tir::Var("v" + std::to_string(count_v++), runtime::DataType::Int(32)),
          tir::IterVarType::kDataPar, ""));
    }
  }
  node->dtype = op->output_dtype(0);
  data_ = node;
}

OpState LayerStateNode::GetOpState(te::Operation op) const {
  CHECK(op_states.count(op))
      << "Can't find op " << op << " in layer state " << layer << ".\n";
  return op_states.at(op);
}

void LayerStateNode::Split(te::Operation op, tir::IterVar iv, PrimExpr factor,
                           tir::IterVar *p_outer, tir::IterVar *p_inner,
                           int *ordinal) {
  CHECK(op_states.count(op)) << "Op not found.\n";
  OpState op_state = op_states.at(op);
  tir::IterVar outer, inner;
  int ord;
  op_state->Split(iv, factor, &outer, &inner, &ord);
  // propagate the split to consumers
  if (iv->iter_type != tir::IterVarType::kCommReduce) {
    CHECK(feed_graph.count(op)) << "Can't split boundary ops.\n";
    for (auto feed : feed_graph.at(op)) {
      CHECK(op_states.count(feed));
      op_states.at(feed)->PropagateSplit(op, ord, outer, inner);
    }
  }
  // for reduce axis, no change for consumers

  if (p_outer)
    *p_outer = outer;
  if (p_inner)
    *p_inner = inner;
  if (ordinal)
    *ordinal = ord;
}

LayerState::LayerState(Layer layer) {
  auto node = make_object<LayerStateNode>();
  node->layer = layer;
  CHECK(!layer->gradients.size())
      << "Please set the gradients after compute transformation.\n";
  Array<te::Operation> ops = layer->GetAllOps();
  for (auto op : ops) {
    node->all_ops.push_back(op);
    node->op_states[op] = OpState(op);
    for (auto inp : op->InputTensors()) {
      if (!node->feed_graph[inp->op].count(op)) {
        node->feed_graph[inp->op].insert(op);
      }
      node->read_graph[op].push_back(inp->op);
    }
  }
  data_ = node;
}

te::Operation LayerStateNode::Transform(
    te::Operation op, Array<tir::Var> spatial_vars,
    Array<PrimExpr> spatial_forward, Array<PrimExpr> spatial_backward,
    Array<tir::Var> reduce_vars, Array<PrimExpr> reduce_forward,
    Array<PrimExpr> reduce_backward, bool explicit_transform) {
  OpState state = GetOpState(op);
  if (explicit_transform) {
    /* strong */
    te::Operation a, b;
    std::tie(a, b) =
        state->Transform(spatial_vars, spatial_forward, spatial_backward,
                         reduce_vars, reduce_forward, reduce_backward);
    OpState upper(a), lower(b);
    int bias = 0, num_ops = (int)all_ops.size();
    for (; bias < num_ops; ++bias) {
      if (all_ops[bias] == op) {
        break;
      }
    }
    CHECK(bias < num_ops) << "Can't find op " << op << ".\n";
    op_states.erase(op);
    feed_graph[op].insert(lower->op);
    read_graph[lower->op].push_back(op);

    op_states[op] = upper;
    op_states[lower->op] = lower;
    all_ops.insert(all_ops.begin() + bias, lower->op);

    std::unordered_set<te::Operation> consumers;
    if (feed_graph.count(op)) {
      consumers = feed_graph.at(op);
      feed_graph.erase(op);
      feed_graph[lower->op] = consumers;
      for (auto c : consumers) {
        CHECK(read_graph.count(c));
        int num_inputs = (int)read_graph.at(c).size();
        for (int i = 0; i < num_inputs; ++i) {
          if (read_graph.at(c)[i] == op) {
            read_graph[c][i] = lower->op;
          }
        }
      }
    }

    return lower->op;
  } else {
    return op;
  }
}

Layer LayerStateNode::MakeCompute(Array<LayerTensor> inputs) {
  // assumption: previous input/output ops are still input/output ops
  //             previous weight ops are still weight ops
  //             previous constant ops are still constant ops
  std::unordered_map<te::Operation, te::Operation> updates;
  int num_inputs = (int)inputs.size();
  Array<te::Tensor> original_inputs = layer->inputs;
  CHECK(num_inputs == (int)(original_inputs.size()))
      << "Input number mismatch.\n";
  for (int i = 0; i < num_inputs; ++i) {
    updates[original_inputs[i]->op] = inputs[i]->tensor->op;
  }

  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (updates.count(op))
      return;

    Array<te::Tensor> new_inputs;
    if (read_graph.count(op)) {
      for (auto inp_op : read_graph.at(op)) {
        helper(inp_op);
        CHECK(updates.count(inp_op));
        new_inputs.push_back(updates.at(inp_op).output(0));
      }
    }

    CHECK(op_states.count(op)) << "Can't find the op " << op << " in states.\n";
    OpState state = op_states.at(op);
    te::Operation new_op = state->MakeCompute(new_inputs);
    updates[op] = new_op;
  };

  Array<te::Operation> new_ops;
  Array<te::Tensor> new_inputs;
  Array<te::Tensor> new_weights;
  Array<te::Tensor> new_const_tensors;
  for (auto op : layer->ops) {
    helper(op);
    CHECK(updates.count(op));
    new_ops.push_back(updates.at(op));
  }
  for (auto inp : layer->inputs) {
    CHECK(updates.count(inp->op));
    new_inputs.push_back(updates.at(inp->op).output(0));
  }
  for (auto w : layer->weights) {
    CHECK(updates.count(w->op));
    new_weights.push_back(updates.at(w->op).output(0));
  }
  for (auto t : layer->const_tensors) {
    CHECK(updates.count(t->op));
    new_const_tensors.push_back(updates.at(t->op).output(0));
  }

  return Layer(layer->name, new_ops, new_inputs, new_weights,
               layer->const_scalars, new_const_tensors, layer->gradients);
}

void BlockStateNode::Split(Layer layer, te::Operation op, tir::IterVar iv,
                           PrimExpr factor, tir::IterVar *p_outer,
                           tir::IterVar *p_inner, int *ordinal) {
  CHECK(layer_states.count(layer));
  layer_states.at(layer)->Split(op, iv, factor, p_outer, p_inner, ordinal);
}

BlockState::BlockState(Block block) {
  auto node = make_object<BlockStateNode>();
  node->block = block;
  Array<Layer> layers = block->GetAllLayers();
  for (auto layer : layers) {
    node->all_layers.push_back(layer);
    node->layer_states[layer] = LayerState(layer);
    for (auto inp : layer->InputTensors()) {
      if (!node->feed_graph[inp->layer].count(layer)) {
        node->feed_graph[inp->layer].insert(layer);
      }
      node->read_graph[layer].push_back(inp->layer);
    }
  }
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.LayerTensor")
    .set_body_typed([](std::string name, Layer layer, te::Tensor tensor,
                       int value_idx) {
      return LayerTensor(name, layer, tensor, value_idx);
    });

TVM_REGISTER_GLOBAL("ditto.LayerTensorHash")
    .set_body_typed([](LayerTensor tensor) -> int64_t {
      return static_cast<int64_t>(std::hash<LayerTensor>()(tensor));
    });

TVM_REGISTER_GLOBAL("ditto.LayerTensorEqual")
    .set_body_method(&LayerTensor::operator==);

TVM_REGISTER_GLOBAL("ditto.Layer")
    .set_body_typed([](std::string name, Array<te::Operation> ops,
                       Array<te::Tensor> inputs, Array<te::Tensor> weights,
                       Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors,
                       Array<te::Tensor> gradients) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors,
                   gradients);
    });

TVM_REGISTER_GLOBAL("ditto.MakeLayer")
    .set_body_typed([](std::string name, Array<te::Operation> ops,
                       Array<te::Tensor> inputs, Array<te::Tensor> weights,
                       Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors,
                       Array<te::Tensor> gradients) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors,
                   gradients);
    });

TVM_REGISTER_GLOBAL("ditto.LayerHash")
    .set_body_typed([](Layer layer) -> int64_t {
      return static_cast<int64_t>(std::hash<Layer>()(layer));
    });

TVM_REGISTER_GLOBAL("ditto.Block")
    .set_body_typed([](std::string name, Array<LayerTensor> out_tensors) {
      return Block(name, out_tensors);
    });

TVM_REGISTER_GLOBAL("ditto.Graph")
    .set_body_typed([](std::string name, Array<Block> block_list) {
      return Graph(name, block_list);
    });

TVM_REGISTER_GLOBAL("ditto.ProduceOutputs")
    .set_body_typed([](Layer layer, Array<LayerTensor> inputs) {
      std::vector<LayerTensor> layer_inputs;
      for (auto inp : inputs) {
        layer_inputs.push_back(inp);
      }
      auto ret = layer.ProduceOutputs(layer_inputs);
      Array<LayerTensor> returns;
      for (auto out : ret) {
        returns.push_back(out);
      }
      return returns;
    });

TVM_REGISTER_GLOBAL("ditto.CreateOpState").set_body_typed([](te::Operation op) {
  return OpState(op);
});

TVM_REGISTER_GLOBAL("ditto.OpStateGetAxis")
    .set_body_typed([](OpState op_state) { return op_state->GetAxis(); });

TVM_REGISTER_GLOBAL("ditto.OpStateGetReduceAxis")
    .set_body_typed([](OpState op_state) { return op_state->GetReduceAxis(); });

TVM_REGISTER_GLOBAL("ditto.OpStateTransform")
    .set_body_typed([](OpState op_state, Array<tir::Var> spatial_vars,
                       Array<PrimExpr> spatial_forward,
                       Array<PrimExpr> spatial_backward,
                       Array<tir::Var> reduce_vars,
                       Array<PrimExpr> reduce_forward,
                       Array<PrimExpr> reduce_backward) {
      te::Operation a, b;
      std::tie(a, b) =
          op_state->Transform(spatial_vars, spatial_forward, spatial_backward,
                              reduce_vars, reduce_forward, reduce_backward);
      return Array<te::Operation>({a, b});
    });

TVM_REGISTER_GLOBAL("ditto.OpStateMakeCompute")
    .set_body_typed([](OpState op_state, Array<te::Tensor> inputs) {
      auto ret = op_state->MakeCompute(inputs);
      return ret;
    });

TVM_REGISTER_GLOBAL("ditto.CreateLayerState").set_body_typed([](Layer layer) {
  return LayerState(layer);
});

TVM_REGISTER_GLOBAL("ditto.LayerStateGetOpState")
    .set_body_typed([](LayerState layer_state, te::Operation op) {
      return layer_state->GetOpState(op);
    });

TVM_REGISTER_GLOBAL("ditto.LayerStateMakeCompute")
    .set_body_typed([](LayerState layer_state, Array<LayerTensor> inputs) {
      auto ret = layer_state->MakeCompute(inputs);
      return ret;
    });

} // namespace graph

} // namespace ditto