#include <auto_compute/state.h>
#include <utils/iter_domain.h>

namespace ditto {

namespace auto_compute {

TVM_REGISTER_NODE_TYPE(TensorStateNode);
TVM_REGISTER_NODE_TYPE(OpStateNode);
TVM_REGISTER_NODE_TYPE(LayerStateNode);
TVM_REGISTER_NODE_TYPE(GraphStateNode);

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

PrimExpr
OpStateNode::ReplaceInputs::VisitExpr_(const tir::ProducerLoadNode *op) {
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

PrimExpr OpStateNode::RemapInput::VisitExpr_(const tir::ProducerLoadNode *op) {
  auto t = runtime::Downcast<te::Tensor>(op->producer);
  if (t.defined() && t->op == original_producer_) {
    Array<PrimExpr> new_indices;
    Map<tir::Var, PrimExpr> original_vars_mapping;
    CHECK(self_->input_tensor_states.count(t->op));
    TensorState state = self_->input_tensor_states.at(t->op);
    int num_old_vars = (int)(original_vars_.size());
    CHECK(num_old_vars == (int)state->access_index.size()) << "Dim mismatch.\n";
    for (int i = 0; i < num_old_vars; ++i) {
      original_vars_mapping.Set(original_vars_[i], state->access_index[i]);
    }
    for (auto var : new_vars_) {
      PrimExpr tmp = var;
      PrimExpr express_by_old = tir::Substitute(tmp, new_vars_mapping_);
      PrimExpr express_by_new =
          tir::Substitute(express_by_old, original_vars_mapping);
      PrimExpr new_index = VisitExpr(express_by_new);
      new_indices.push_back(new_index);
    }
    return tir::ProducerLoad(new_producer_.output(0), new_indices);
  } else {
    return tir::ExprMutator::VisitExpr_(op);
  }
}

Array<tir::IterVar> OpStateNode::GetAxis() const {
  return Array<tir::IterVar>(axis);
}

Array<tir::IterVar> OpStateNode::GetReduceAxis() const {
  return Array<tir::IterVar>(reduce_axis);
}

std::pair<te::Operation, te::Operation> OpStateNode::TransformIsolation(
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
    // std::unordered_map<const tir::VarNode *, tir::IterVarType> new_types =
    //     utils::InferIterVarType(var_mapping, original_types);

    Array<tir::IterVar> new_itervars;
    for (auto var : new_vars) {
      CHECK(new_var_mapping.count(var))
          << "Can't find the var " << var << " after infer range.\n";
      // CHECK(new_types.count(var.get()))
      //     << "Can't find the var " << var << " after infer type.\n";
      // CHECK(new_types.at(var.get()) == type)
      //     << "Expect type " << type << " but get " << new_types.at(var.get())
      //     << ".\n";
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
      PrimExpr body = tir::Substitute(b, reverse_mapping);
      const tir::ReduceNode *red = body.as<tir::ReduceNode>();
      if (red) {
        // reduce is always top-level
        // change the reduce axis
        body =
            tir::Reduce(red->combiner, red->source, upper_reduce_axis,
                        red->condition, red->value_index, red->init, red->span);
      }
      new_body.push_back(body);
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

te::Operation OpStateNode::TransformRemapInput(
    int original_producer_location, te::Operation new_producer,
    Array<tir::Var> original_vars, Array<tir::Var> new_vars,
    Map<tir::Var, PrimExpr> new_vars_mapping) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop) << "Only expect ComputeOp to have inputs.\n";
  Array<PrimExpr> new_body;
  // the producer op is not sync with the tag we use in layer state
  Array<te::Tensor> inputs = op->InputTensors();
  CHECK(0 <= original_producer_location &&
        original_producer_location < (int)(inputs.size()))
      << "Input location out of bound.\n";
  te::Operation original_producer = inputs[original_producer_location]->op;
  OpStateNode::RemapInput remap(this, original_producer, new_producer,
                                original_vars, new_vars, new_vars_mapping);
  for (auto b : cop->body) {
    PrimExpr new_b = remap(b);
    new_body.push_back(new_b);
  }
  return te::ComputeOp(cop->name, cop->tag, cop->attrs, axis, new_body);
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
  ReplaceInputs mutator(this, mapping);
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

te::Operation LayerStateNode::Transform(
    te::Operation op, Array<tir::Var> spatial_vars,
    Array<PrimExpr> spatial_forward, Array<PrimExpr> spatial_backward,
    Array<tir::Var> reduce_vars, Array<PrimExpr> reduce_forward,
    Array<PrimExpr> reduce_backward, bool explicit_transform) {
  OpState state = GetOpState(op);
  /*
  // one op is transformed according to the given transformations
  // Example:
  //      O
  //      |
  //      O  <- op to transform
  //     / \
  //     O O
  /////////////////////////////////////
  //      |
  //      v
  /////////////////////////////////////
  // Explicit Transform
  //      O
  //      |
  //      O  <- op that inherit origianl op
  //      |
  //      O  <- new op to correct the data (return value)
  //     / \
  //     O O
  /////////////////////////////////////
  // Implicit Transform
  //      O
  //      |
  //      O  <- op that inherit origianl op (return value)
  //     / \
  //     O O  <- these consumers should be remapped
  */
  te::Operation a, b;
  std::tie(a, b) =
      state->TransformIsolation(spatial_vars, spatial_forward, spatial_backward,
                                reduce_vars, reduce_forward, reduce_backward);
  // upper: op that inherit original op
  // lower: correction op
  OpState upper(a), lower(b);
  int bias = 0, num_ops = (int)all_ops.size();
  // find the current op
  for (; bias < num_ops; ++bias) {
    if (all_ops[bias] == op) {
      break;
    }
  }
  CHECK(bias < num_ops) << "Can't find op " << op << ".\n";
  if (explicit_transform) { // explicit
    op_states.erase(op);
    read_graph[lower->op].push_back(op); // we still use original op as tag

    op_states[op] = upper; // we still use original op as tag
    op_states[lower->op] = lower;
    // keep the topological order: left inputs, right outputs
    all_ops.insert(all_ops.begin() + bias, lower->op);

    std::unordered_set<te::Operation> consumers;
    if (feed_graph.count(op)) {
      consumers = feed_graph.at(op);
      // delete original outputs and inherit to lower op
      feed_graph.erase(op);
      feed_graph[lower->op] = consumers;
      // modify the consumers' input info
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
    // add the output of upper op
    feed_graph[op].insert(lower->op);

    return lower->op;
  } else { // implicit
    op_states.erase(op);
    op_states[op] = upper;

    std::unordered_set<te::Operation> consumers;
    if (feed_graph.count(op)) {
      consumers = feed_graph.at(op);

      for (auto c : consumers) {
        CHECK(op_states.count(c));
        OpState old_state = op_states.at(c);
        CHECK(read_graph.count(c));
        int num_inputs = (int)read_graph.at(c).size();
        // record the first occur of current op in the inptus
        int original_producer_location = num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          if (read_graph.at(c)[i] == op) {
            original_producer_location = i;
            break;
          }
        }
        CHECK(original_producer_location < num_inputs);
        te::Operation new_producer = upper->op; // this seems not important
        Array<tir::Var> original_vars;
        Array<tir::Var> new_vars = spatial_vars;
        Map<tir::Var, PrimExpr> new_vars_mapping;
        int num_axis = (int)(state->axis.size());
        for (int i = 0; i < num_axis; ++i) {
          original_vars.push_back(state->axis[i]->var);
        }
        int num_new_axis = (int)(spatial_vars.size());
        for (int i = 0; i < num_new_axis; ++i) {
          new_vars_mapping.Set(spatial_vars[i], spatial_forward[i]);
        }
        te::Operation new_c = old_state->TransformRemapInput(
            original_producer_location, new_producer, original_vars, new_vars,
            new_vars_mapping);

        OpState new_state(new_c);
        op_states[c] = new_state;
      }
    }

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
               layer->const_scalars, new_const_tensors);
}

Array<te::Operation> LayerStateNode::GetCurrentOps() {
  return Array<te::Operation>(all_ops);
}

LayerState::LayerState(Layer layer) {
  auto node = make_object<LayerStateNode>();
  node->layer = layer;
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

// BlockState::BlockState(Block block) {
//   auto node = make_object<BlockStateNode>();
//   node->block = block;
//   Array<Layer> layers = block->GetAllLayers();
//   for (auto layer : layers) {
//     node->all_layers.push_back(layer);
//     node->layer_states[layer] = LayerState(layer);
//     for (auto inp : layer->InputTensors()) {
//       if (!node->feed_graph[inp->layer].count(layer)) {
//         node->feed_graph[inp->layer].insert(layer);
//       }
//       node->read_graph[layer].push_back(inp->layer);
//     }
//   }
//   data_ = node;
// }

LayerState GraphStateNode::GetLayerState(Layer layer) const {
  CHECK(layer_states.count(layer))
      << "Can't find layer " << layer << " in graph state " << graph << ".\n";
  return layer_states.at(layer);
}

Graph GraphStateNode::MakeCompute(Array<LayerTensor> inputs) {
  std::unordered_map<LayerTensor, LayerTensor> updates;
  int num_inputs = (int)inputs.size();
  Array<LayerTensor> original_inputs = graph->graph_inputs;
  CHECK(num_inputs == (int)(original_inputs.size()))
      << "Input number mismatch.\n";
  for (int i = 0; i < num_inputs; ++i) {
    updates[original_inputs[i]] = inputs[i];
  }

  std::function<void(LayerTensor lt)> helper;
  helper = [&](LayerTensor lt) {
    if (updates.count(lt))
      return;

    std::vector<LayerTensor> new_inputs;
    if (lt->layer.defined() && consume_graph.count(lt->layer)) {
      for (auto inp_lt : consume_graph.at(lt->layer)) {
        helper(inp_lt);
        CHECK(updates.count(inp_lt));
        new_inputs.push_back(updates.at(inp_lt));
      }
    }

    if (!lt->layer.defined()) {
      updates[lt] = lt;
    } else {
      CHECK(layer_states.count(lt->layer))
          << "Can't find the layer " << lt->layer->name << " in states.\n";
      CHECK(produce_graph.count(lt->layer))
          << "Can't find the layer " << lt->layer->name
          << " in produce graph.\n";
      LayerState state = layer_states.at(lt->layer);
      Layer new_layer = state->MakeCompute(Array<LayerTensor>(new_inputs));
      Array<LayerTensor> new_outputs = new_layer.ProduceOutputs(new_inputs);
      Array<LayerTensor> old_outputs = produce_graph.at(lt->layer);
      int num_outputs = (int)new_outputs.size();
      CHECK(num_outputs == (int)old_outputs.size())
          << "Output number mismatch.\n";
      for (int i = 0; i < num_outputs; ++i) {
        LayerTensor tmp = new_outputs[i];
        te::Tensor p = te::placeholder(tmp->tensor->shape, tmp->tensor->dtype,
                                       tmp->tensor->op->name);
        updates[old_outputs[i]] =
            LayerTensor(tmp->name, tmp->layer, p, tmp->value_idx);
      }
    }
  };

  Array<LayerTensor> new_outputs;
  Array<LayerTensor> new_inputs;
  for (auto lt : this->outputs) {
    helper(lt);
    CHECK(updates.count(lt));
    new_outputs.push_back(updates.at(lt));
  }
  for (auto lt : this->inputs) {
    CHECK(updates.count(lt));
    new_inputs.push_back(updates.at(lt));
  }

  return Graph(graph->name, new_inputs, new_outputs);
}

Array<Layer> GraphStateNode::GetCurrentLayers() {
  return Array<Layer>(all_layers);
}

Array<Layer> GraphStateNode::NormalizePartition(Layer layer, bool modify) {
  LayerState state = this->GetLayerState(layer);
  CHECK(this->consume_graph.count(layer));
  Layer new_layer =
      state->MakeCompute(Array<LayerTensor>(this->consume_graph.at(layer)));
  LayerState new_state(new_layer);
  int layer_pos = 0;
  for (auto l : this->all_layers) {
    if (l == layer) {
      break;
    }
    layer_pos += 1;
  }
  CHECK(layer_pos < (int)this->all_layers.size());

  // check the scalar is empty
  CHECK(new_layer->const_scalars.size() == 0U)
      << "Const scalar feature is not enabled now, please do not use this "
         "attribute.\n";

  std::unordered_map<te::Operation, Layer> op2layer;
  std::unordered_map<te::Operation, int> op2outputs;
  std::unordered_map<te::Operation, int> op2inputs;
  std::unordered_map<te::Operation, int> op2weights;
  std::unordered_map<te::Operation, int> op2const_tensors;

  int num_outputs = (int)new_layer->ops.size();
  for (int i = 0; i < num_outputs; ++i) {
    op2outputs[new_layer->ops[i]] = i;
  }
  int num_inputs = (int)new_layer->inputs.size();
  for (int i = 0; i < num_inputs; ++i) {
    op2inputs[new_layer->inputs[i]->op] = i;
  }
  int num_weights = (int)new_layer->weights.size();
  for (int i = 0; i < num_weights; ++i) {
    op2weights[new_layer->weights[i]->op] = i;
  }
  int num_const_tensors = (int)new_layer->const_tensors.size();
  for (int i = 0; i < num_const_tensors; ++i) {
    op2const_tensors[new_layer->const_tensors[i]->op] = i;
  }

  std::unordered_map<te::Operation, std::vector<LayerTensor>>
      update_consume_graph;
  std::unordered_map<te::Operation, std::vector<LayerTensor>>
      update_produce_graph;
  std::unordered_map<LayerTensor, LayerTensor> update_layer_outputs;
  std::unordered_map<int, std::vector<std::pair<te::Operation, int>>>
      update_feed_graph;

  int count_sub_layer = 0;
  Array<Layer> ret;
  for (auto op : new_state->all_ops) {
    CHECK(new_state->op_states.count(op));
    CHECK(!op2layer.count(op));
    // op is up-to-date
    Array<te::Operation> ops;
    Array<te::Tensor> inputs;
    Array<te::Tensor> weights;
    Array<PrimExpr> const_scalars;
    Array<te::Tensor> const_tensors;

    const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    CHECK(pop || cop);
    if (pop != nullptr) {
      continue;
    } else {
      // cop
      std::unordered_map<te::Operation, te::Operation> tmp_map;

      if (new_state->read_graph.count(op)) {
        for (auto iop : new_state->read_graph.at(op)) {
          if (op2layer.count(iop)) {
            // read from another new layer's output
            Layer this_layer = op2layer.at(iop);
            CHECK(this_layer->ops.size() > 0U);
            te::Tensor out = this_layer->ops[0].output(0);
            te::Tensor tmp =
                te::placeholder(out->shape, out->dtype, out->op->name);
            LayerTensor lt(out->op->name, this_layer, tmp, out->value_index);
            inputs.push_back(tmp);
            update_consume_graph[op].push_back(lt);
            tmp_map[iop] = tmp->op;
          } else if (op2inputs.count(iop)) {
            // read from the layer's inputs
            CHECK(this->consume_graph.count(layer));
            Array<LayerTensor> input_lts = this->consume_graph.at(layer);
            int idx = op2inputs.at(iop);

            LayerTensor lt = input_lts[idx];

            // CHECK(lt->tensor->op == iop);
            inputs.push_back(iop.output(0));
            update_consume_graph[op].push_back(lt);
            update_feed_graph[idx].push_back(
                std::make_pair(op, (int)inputs.size() - 1));

            tmp_map[iop] = iop;
          } else if (op2weights.count(iop)) {
            // read from the layer's weights
            weights.push_back(iop.output(0));
            tmp_map[iop] = iop;
          } else if (op2const_tensors.count(iop)) {
            // read from the layer's const tensors
            const_tensors.push_back(iop.output(0));
            tmp_map[iop] = iop;
          } else {
            CHECK(false) << "Unknown source for input " << iop << ".\n";
          }
        }
      }

      OpState tmp_state(op);
      Array<te::Tensor> op_inputs;
      for (auto inp : op->InputTensors()) {
        CHECK(tmp_map.count(inp->op));
        op_inputs.push_back(tmp_map.at(inp->op).output(0));
      }
      te::Operation new_op = tmp_state->MakeCompute(op_inputs);
      ops.push_back(new_op);
      Layer cop_layer =
          Layer(new_layer->name + "_" + std::to_string(count_sub_layer++), ops,
                inputs, weights, const_scalars, const_tensors);

      std::vector<LayerTensor> cop_inputs;
      if (update_consume_graph.count(op)) {
        for (auto cop_inp : update_consume_graph.at(op)) {
          cop_inputs.push_back(cop_inp);
        }
      }
      std::vector<LayerTensor> cop_outputs =
          cop_layer.ProduceOutputs(cop_inputs);
      update_produce_graph[op] = cop_outputs;

      if (op2outputs.count(op)) {
        int idx = op2outputs.at(op);
        CHECK(this->produce_graph.count(layer));
        CHECK((int)this->produce_graph.at(layer).size() > idx);
        CHECK(cop_outputs.size() == 1U);
        update_layer_outputs[this->produce_graph.at(layer)[idx]] =
            cop_outputs[0];
      }

      op2layer[op] = cop_layer;
      ret.push_back(cop_layer);
    }
  }

  if (!modify) {
    return ret;
  }
  // update the graph state
  // update all_layers
  this->all_layers.erase(this->all_layers.begin() + layer_pos);
  for (int i = count_sub_layer - 1; i >= 0; --i) {
    this->all_layers.insert(this->all_layers.begin() + layer_pos, ret[i]);
    this->layer_states[ret[i]] = LayerState(ret[i]);
  }
  // update consume_graph, produce_graph, feed_graph
  for (auto op : new_state->all_ops) {
    if (op2layer.count(op)) {
      Layer l = op2layer.at(op);
      this->consume_graph[l] = update_consume_graph[op];
      this->produce_graph[l] = update_produce_graph[op];
    }
  }

  if (this->produce_graph.count(layer)) {
    for (auto plt : produce_graph.at(layer)) {
      if (this->feed_graph.count(plt)) {
        for (auto kv : this->feed_graph.at(plt)) {
          CHECK(this->consume_graph.count(kv.first));
          CHECK((int)this->consume_graph.at(kv.first).size() > kv.second);
          CHECK(update_layer_outputs.count(plt));
          CHECK(update_layer_outputs.at(plt) != plt);
          LayerTensor output_to_add = update_layer_outputs.at(plt);
          te::Tensor tmp = te::placeholder(output_to_add->tensor->shape,
                                           output_to_add->tensor->dtype,
                                           output_to_add->tensor->op->name);
          LayerTensor new_output =
              LayerTensor(output_to_add->name, output_to_add->layer, tmp,
                          output_to_add->value_idx);
          this->consume_graph[kv.first][kv.second] = new_output;
          this->feed_graph[update_layer_outputs.at(plt)].push_back(kv);
        }
      }
    }
  }

  // update feed_graph
  if (this->consume_graph.count(layer)) {
    for (auto inp : this->consume_graph.at(layer)) {
      CHECK(this->feed_graph.count(inp));
      std::vector<std::pair<Layer, int>> update;
      for (auto kv : this->feed_graph.at(inp)) {
        if (kv.first != layer) {
          update.push_back(kv);
        } else {
          CHECK(update_feed_graph.count(kv.second));
          for (auto kkvv : update_feed_graph.at(kv.second)) {
            CHECK(op2layer.count(kkvv.first));
            Layer l = op2layer.at(kkvv.first);
            update.push_back(std::make_pair(l, kkvv.second));
          }
        }
      }
      this->feed_graph[inp] = update;
    }
  }

  for (auto op : new_state->all_ops) {
    if (op2layer.count(op)) {
      if (new_state->feed_graph.count(op)) {
        for (auto fop : new_state->feed_graph.at(op)) {
          std::vector<int> pos;
          int count = 0;
          for (auto inp : fop->InputTensors()) {
            if (inp->op == op) {
              pos.push_back(count);
            }
            count += 1;
          }

          CHECK(op2layer.count(fop));
          CHECK(update_produce_graph.count(op));
          std::vector<LayerTensor> o = update_produce_graph.at(op);
          CHECK(o.size() == 1U);
          for (auto p : pos) {
            this->feed_graph[o[0]].push_back(
                std::make_pair(op2layer.at(fop), p));
          }
        }
      }
    }
  }

  // erase old info
  if (this->consume_graph.count(layer)) {
    this->consume_graph.erase(layer);
  }
  if (this->produce_graph.count(layer)) {
    this->produce_graph.erase(layer);
  }
  for (auto kv : update_layer_outputs) {
    if (this->feed_graph.count(kv.first)) {
      this->feed_graph.erase(kv.first);
    }
  }

  // update outputs
  std::vector<LayerTensor> new_outputs;
  for (auto l : this->outputs) {
    if (update_layer_outputs.count(l)) {
      new_outputs.push_back(update_layer_outputs.at(l));
    } else {
      new_outputs.push_back(l);
    }
  }
  this->outputs = new_outputs;

  return ret;
}

Layer GraphStateNode::Fuse(Layer front, Layer back, bool modify) {
  Array<Layer> source;
  Array<Layer> sink;
  source.push_back(front);
  sink.push_back(back);
  Array<Layer> convex_set = FindConvexSet(source, sink);
  CHECK(convex_set.size() > 0U) << "The fusion set is empty.\n";
  std::unordered_set<Layer> layer_convex_set;
  for (auto layer : convex_set) {
    layer_convex_set.insert(layer);
  }

  std::unordered_set<LayerTensor> graph_output_set;
  for (auto out : this->outputs) {
    graph_output_set.insert(out);
  }

  // ctx for layers
  // use old layer as tag
  std::unordered_map<Layer, std::unordered_map<te::Tensor, int>> inputs_map;
  std::unordered_map<Layer, std::unordered_map<te::Tensor, int>> weights_map;
  std::unordered_map<Layer, std::unordered_map<te::Tensor, int>>
      const_tensors_map;

  // info to build final layer
  Array<te::Operation> ops;
  Array<te::Tensor> inputs;
  Array<te::Tensor> weights;
  Array<PrimExpr> const_scalars;
  Array<te::Tensor> const_tensors;
  std::vector<LayerTensor> fused_layer_inputs;

  std::unordered_map<LayerTensor, std::vector<int>> fuse_feed_graph;
  std::unordered_set<LayerTensor> eliminated_layer_tensor;
  std::unordered_map<LayerTensor, int> update_layer_tensor;

  std::unordered_map<Layer, Layer> update_layer;
  std::unordered_set<te::Operation> visit_op;
  std::unordered_map<te::Operation, te::Operation> update_op;
  std::function<void(Layer)> fuse_from_layer;
  std::function<void(Layer, te::Operation)> fuse_from_op;
  fuse_from_op = [&](Layer env, te::Operation op) {
    if (!op.defined() || visit_op.count(op)) {
      return;
    }
    visit_op.insert(op);
    const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    CHECK(pop || cop);
    if (pop) {
      if (inputs_map.count(env) && inputs_map[env].count(op.output(0))) {
        int idx = inputs_map[env][op.output(0)];
        CHECK(update_layer.count(env));
        Layer new_env = update_layer.at(env);
        CHECK(this->consume_graph.count(env));
        CHECK((int)this->consume_graph.at(env).size() > idx);
        LayerTensor lt = this->consume_graph.at(env)[idx];
        CHECK(lt->tensor == new_env->inputs[idx]);
        if (!lt->layer.defined()) {
          // update inputs for fused layer
          inputs.push_back(lt->tensor);
          fused_layer_inputs.push_back(lt);
          fuse_feed_graph[lt].push_back((int)inputs.size() - 1);
          update_op[op] = op;
        } else if (!layer_convex_set.count(lt->layer)) {
          // update inputs for fused layer
          inputs.push_back(lt->tensor);
          fused_layer_inputs.push_back(lt);
          fuse_feed_graph[lt].push_back((int)inputs.size() - 1);
          update_op[op] = op;
        } else {
          fuse_from_layer(lt->layer);
          CHECK(update_layer.count(lt->layer));
          Layer new_lt_layer = update_layer.at(lt->layer);
          CHECK((int)new_lt_layer->ops.size() > lt->value_idx);
          te::Operation old_op = new_lt_layer->ops[lt->value_idx];
          CHECK(update_op.count(old_op));
          te::Operation new_op = update_op.at(old_op);
          update_op[op] = new_op;
        }
      } else if (weights_map.count(env) &&
                 weights_map[env].count(op.output(0))) {
        int idx = weights_map[env][op.output(0)];
        CHECK(update_layer.count(env));
        Layer new_env = update_layer.at(env);
        CHECK((int)new_env->weights.size() > idx);
        // update weights for fused layer
        weights.push_back(op.output(0));
        update_op[op] = op;
      } else if (const_tensors_map.count(env) &&
                 const_tensors_map[env].count(op.output(0))) {
        int idx = const_tensors_map[env][op.output(0)];
        CHECK(update_layer.count(env));
        Layer new_env = update_layer.at(env);
        CHECK((int)new_env->const_tensors.size() > idx);
        // update const_tensors for fused layer
        const_tensors.push_back(op.output(0));
        update_op[op] = op;
      } else {
        CHECK(false) << "Unknown source for op " << op << " form layer "
                     << env->name << ".\n";
      }
    } else {
      // cop
      Array<te::Tensor> new_inputs;
      for (auto inp : cop->InputTensors()) {
        fuse_from_op(env, inp->op);
        CHECK(update_op.count(inp->op));
        te::Operation new_inp_op = update_op.at(inp->op);
        new_inputs.push_back(new_inp_op.output(0));
      }
      OpState tmp_state(op);
      te::Operation new_op = tmp_state->MakeCompute(new_inputs);
      update_op[op] = new_op;
    }
  };

  fuse_from_layer = [&](Layer layer) {
    if (!layer.defined() || update_layer.count(layer)) {
      return;
    }
    LayerState state = this->GetLayerState(layer);
    CHECK(this->consume_graph.count(layer));
    std::vector<LayerTensor> tmp_inputs = this->consume_graph.at(layer);
    Layer new_layer = state->MakeCompute(Array<LayerTensor>(tmp_inputs));
    new_layer.ProduceOutputs(tmp_inputs);
    update_layer[layer] = new_layer;
    // fill ctx for layer
    int num_inputs = (int)new_layer->inputs.size();
    for (int i = 0; i < num_inputs; ++i) {
      inputs_map[layer][new_layer->inputs[i]] = i;
    }
    int num_weights = (int)new_layer->weights.size();
    for (int i = 0; i < num_weights; ++i) {
      weights_map[layer][new_layer->weights[i]] = i;
    }
    int num_const_tensors = (int)new_layer->const_tensors.size();
    for (int i = 0; i < num_const_tensors; ++i) {
      const_tensors_map[layer][new_layer->const_tensors[i]] = i;
    }
    CHECK(new_layer->const_scalars.size() == 0U)
        << "Const scalar feature is not enabled now, please do not use this "
        << "attribute.\n";
    int count_output = 0;
    CHECK(this->produce_graph.count(layer));
    std::vector<LayerTensor> tmp_outputs = this->produce_graph.at(layer);
    for (auto op : new_layer->ops) {
      fuse_from_op(layer, op);
      CHECK(update_op.count(op));
      te::Operation new_op = update_op.at(op);
      LayerTensor old_lt = tmp_outputs[count_output];
      bool eliminate = true;
      // check whether to add to ops
      if (this->feed_graph.count(old_lt)) {
        // has consumers
        for (auto kv : this->feed_graph.at(old_lt)) {
          if (!layer_convex_set.count(kv.first)) {
            // consumed by layers outside the convex set
            eliminate = false;
            break;
          }
        }
      } else if (graph_output_set.count(old_lt)) {
        // graph output
        eliminate = false;
      }
      if (eliminate) {
        eliminated_layer_tensor.insert(old_lt);
      } else {
        // update ops for fused layer
        ops.push_back(new_op);
        // store the position of output
        update_layer_tensor[old_lt] = (int)ops.size() - 1;
      }
      count_output += 1;
    }
  };

  // execute the fuse process
  fuse_from_layer(back);

  std::string name = "fused";
  for (auto l : convex_set) {
    name = name + "." + l->name;
  }
  Layer fused(name, ops, inputs, weights, const_scalars, const_tensors);
  if (!modify) {
    return fused;
  }

  // update context
  // update produce, consume graph, feed graph
  std::vector<LayerTensor> fused_layer_outputs =
      fused.ProduceOutputs(fused_layer_inputs);
  this->produce_graph[fused] = fused_layer_outputs;
  this->consume_graph[fused] = fused_layer_inputs;
  int num_outputs = (int)fused_layer_outputs.size();
  for (auto kv : update_layer_tensor) {
    CHECK(kv.second < num_outputs);
    if (this->feed_graph.count(kv.first)) {
      for (auto kkvv : this->feed_graph.at(kv.first)) {
        if (!layer_convex_set.count(kkvv.first)) {
          CHECK(this->consume_graph.count(kkvv.first));
          CHECK((int)this->consume_graph.at(kkvv.first).size() > kkvv.second);
          LayerTensor output_to_add = fused_layer_outputs[kv.second];
          te::Tensor tmp = te::placeholder(output_to_add->tensor->shape,
                                           output_to_add->tensor->dtype,
                                           output_to_add->tensor->op->name);
          LayerTensor new_output =
              LayerTensor(output_to_add->name, output_to_add->layer, tmp,
                          output_to_add->value_idx);
          this->consume_graph.at(kkvv.first)[kkvv.second] = new_output;
          this->feed_graph[fused_layer_outputs[kv.second]].push_back(kkvv);
        }
      }
    }
  }

  for (auto kv : fuse_feed_graph) {
    for (auto v : kv.second) {
      this->feed_graph[kv.first].push_back(std::make_pair(fused, v));
    }
  }

  // update graph outputs
  std::vector<LayerTensor> new_outputs;
  for (auto lt : this->outputs) {
    if (update_layer_tensor.count(lt)) {
      int idx = update_layer_tensor.at(lt);
      new_outputs.push_back(fused_layer_outputs[idx]);
    } else {
      new_outputs.push_back(lt);
    }
  }
  this->outputs = new_outputs;

  // delete out-of-data feed graph
  for (auto kv : update_layer_tensor) {
    if (this->feed_graph.count(kv.first)) {
      this->feed_graph.erase(kv.first);
    }
  }
  for (auto lt : eliminated_layer_tensor) {
    if (this->feed_graph.count(lt)) {
      this->feed_graph.erase(lt);
    }
  }

  std::vector<Layer> new_layers;
  for (auto l : this->all_layers) {
    if (layer_convex_set.count(l)) {
      if (l == front) {
        new_layers.push_back(fused);
      }
      this->layer_states.erase(l);
      if (this->consume_graph.count(l)) {
        for (auto inp : this->consume_graph.at(l)) {
          if (this->feed_graph.count(inp)) {
            std::vector<std::pair<Layer, int>> update;
            for (auto kv : this->feed_graph.at(inp)) {
              if (kv.first != l) {
                update.push_back(kv);
              }
            }
            if (update.size() > 0U) {
              this->feed_graph[inp] = update;
            } else {
              this->feed_graph.erase(inp);
            }
          }
        }
        this->consume_graph.erase(l);
      }
      if (this->produce_graph.count(l)) {
        this->produce_graph.erase(l);
      }
    } else {
      new_layers.push_back(l);
    }
  }
  this->all_layers = new_layers;
  LayerState fused_state(fused);
  this->layer_states[fused] = fused_state;

  return fused;
}

GraphState::GraphState(Graph graph) {
  auto node = make_object<GraphStateNode>();
  node->graph = graph;
  for (auto l : graph->graph_inputs) {
    node->inputs.push_back(l);
  }
  for (auto l : graph->graph_outputs) {
    node->outputs.push_back(l);
  }
  Array<Layer> layers = graph->GetAllLayers();
  for (auto layer : layers) {
    node->all_layers.push_back(layer);
    node->layer_states[layer] = LayerState(layer);
    int count_inp = 0;
    for (auto inp : layer->input_layer_tensors_) {
      node->consume_graph[layer].push_back(inp);
      node->feed_graph[inp].push_back(std::make_pair(layer, count_inp));
      count_inp += 1;
    }
    for (auto out : layer->output_layer_tensors_) {
      node->produce_graph[layer].push_back(out);
    }
  }
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_compute.CreateOpState")
    .set_body_typed([](te::Operation op) { return OpState(op); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.OpStateGetAxis")
    .set_body_typed([](OpState op_state) { return op_state->GetAxis(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.OpStateGetReduceAxis")
    .set_body_typed([](OpState op_state) { return op_state->GetReduceAxis(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.OpStateTransform")
    .set_body_typed([](OpState op_state, Array<tir::Var> spatial_vars,
                       Array<PrimExpr> spatial_forward,
                       Array<PrimExpr> spatial_backward,
                       Array<tir::Var> reduce_vars,
                       Array<PrimExpr> reduce_forward,
                       Array<PrimExpr> reduce_backward) {
      te::Operation a, b;
      std::tie(a, b) = op_state->TransformIsolation(
          spatial_vars, spatial_forward, spatial_backward, reduce_vars,
          reduce_forward, reduce_backward);
      return Array<te::Operation>({a, b});
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.OpStateMakeCompute")
    .set_body_typed([](OpState op_state, Array<te::Tensor> inputs) {
      auto ret = op_state->MakeCompute(inputs);
      return ret;
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.CreateLayerState")
    .set_body_typed([](Layer layer) { return LayerState(layer); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerStateGetOpState")
    .set_body_typed([](LayerState layer_state, te::Operation op) {
      return layer_state->GetOpState(op);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerStateTransform")
    .set_body_typed(
        [](LayerState layer_state, te::Operation op,
           Array<tir::Var> spatial_vars, Array<PrimExpr> spatial_forward,
           Array<PrimExpr> spatial_backward, Array<tir::Var> reduce_vars,
           Array<PrimExpr> reduce_forward, Array<PrimExpr> reduce_backward,
           int explicit_transform) {
          return layer_state->Transform(
              op, spatial_vars, spatial_forward, spatial_backward, reduce_vars,
              reduce_forward, reduce_backward, explicit_transform == 1);
        });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerStateMakeCompute")
    .set_body_typed([](LayerState layer_state, Array<LayerTensor> inputs) {
      auto ret = layer_state->MakeCompute(inputs);
      return ret;
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerStateGetCurrentOps")
    .set_body_typed([](LayerState layer_state) {
      return layer_state->GetCurrentOps();
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.CreateGraphState")
    .set_body_typed([](Graph graph) { return GraphState(graph); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphStateGetLayerState")
    .set_body_typed([](GraphState graph_state, Layer layer) {
      return graph_state->GetLayerState(layer);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphStateMakeCompute")
    .set_body_typed([](GraphState graph_state, Array<LayerTensor> inputs) {
      auto ret = graph_state->MakeCompute(inputs);
      return ret;
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphStateGetCurrentLayers")
    .set_body_typed([](GraphState graph_state) {
      return graph_state->GetCurrentLayers();
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphStateNormalizePartitionLayer")
    .set_body_typed([](GraphState graph_state, Layer layer, bool modify) {
      return graph_state->NormalizePartition(layer, modify);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphStateFuseLayer")
    .set_body_typed([](GraphState graph_state, Layer front, Layer back,
                       bool modify) {
      return graph_state->Fuse(front, back, modify);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.FindConvexSet")
    .set_body_typed([](Layer front, Layer back) {
      Array<Layer> source;
      Array<Layer> sink;
      source.push_back(front);
      sink.push_back(back);
      return FindConvexSet(source, sink);
    });

} // namespace auto_compute

} // namespace ditto