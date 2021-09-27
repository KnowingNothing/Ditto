#include <auto_compute/state.h>
#include <utils/iter_domain.h>

namespace ditto {

namespace auto_compute {

TVM_REGISTER_NODE_TYPE(TensorStateNode);
TVM_REGISTER_NODE_TYPE(OpStateNode);
TVM_REGISTER_NODE_TYPE(LayerStateNode);
TVM_REGISTER_NODE_TYPE(BlockStateNode);

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
               layer->const_scalars, new_const_tensors, layer->gradients);
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

TVM_REGISTER_GLOBAL("ditto.auto_compute.CreateOpState").set_body_typed([](te::Operation op) {
  return OpState(op);
});

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

TVM_REGISTER_GLOBAL("ditto.auto_compute.CreateLayerState").set_body_typed([](Layer layer) {
  return LayerState(layer);
});

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

} // namespace auto_compute

} // namespace ditto