#include <auto_compute/graph.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/op.h>
#include <utils/fingerprint.h>
#include <utils/iter_domain.h>

#include <deque>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

using namespace tvm;
namespace ditto {

namespace auto_compute {

TVM_REGISTER_NODE_TYPE(LayerTensorNode);
TVM_REGISTER_NODE_TYPE(LayerNode);
TVM_REGISTER_NODE_TYPE(GraphNode);

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
  std::vector<te::Operation> ret;

  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (visit.count(op))
      return;
    visit.insert(op);
    for (auto inp : op->InputTensors()) {
      helper(inp->op);
    }
    ret.push_back(op);
  };

  for (auto op : ops) {
    helper(op);
  }

  // left to right: inputs to outputs
  // std::reverse(ret.begin(), ret.end());
  return Array<te::Operation>(ret);
}

std::string LayerNode::GetFingerprint() const {
  std::ostringstream oss;
  oss.str("");
  oss << "Layer(\n";
  Array<te::Operation> all_ops = this->GetAllOps();
  int count_op = 0;
  std::unordered_map<te::Operation, std::string> op_rename;
  oss << "ops=\n";
  for (auto op : all_ops) {
    std::string op_name = "Op" + std::to_string(count_op++);
    op_rename[op] = op_name;
    const te::PlaceholderOpNode *pop = op.as<te::PlaceholderOpNode>();
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    CHECK(pop || cop) << "Only expect placeholder or compute op.\n";
    if (pop) {
      oss << op_name << "(" << op.output(0)->shape << ")\n";
    } else {
      // cop
      oss << op_name << "(\n"
          << utils::GetFingerPrint(cop->axis, cop->body) << ")\n";
    }
    oss << "read_graph=(";
    bool is_first = true;
    for (auto inp : op->InputTensors()) {
      CHECK(op_rename.count(inp->op));
      if (is_first) {
        is_first = false;
      } else {
        oss << ", ";
      }
      oss << op_rename.at(inp->op);
    }
    oss << ")\n";
  }
  oss << ")\n";
  return oss.str();
}

FloatImm LayerNode::GetDataTransferAmount() const {
  float ret = 0;
  for (auto op : this->GetAllOps()) {
    for (auto inp : op->InputTensors()) {
      int ele = 1;
      for (auto s : inp->shape) {
        const IntImmNode *as_int = s.as<IntImmNode>();
        CHECK(as_int) << "Please use static shape, rather than " << s << ".\n";
        ele = ele * as_int->value;
      }
      ret += ele;
    }
  }
  for (auto op : this->ops) {
    int ele = 1;
    for (auto s : op.output(0)->shape) {
      const IntImmNode *as_int = s.as<IntImmNode>();
      CHECK(as_int) << "Please use static shape, rather than " << s << ".\n";
      ele = ele * as_int->value;
    }
    ret += ele;
  }
  return FloatImm(tvm::runtime::DataType::Float(32), ret);
}

FloatImm LayerNode::GetGFLOPS() const {
  float ret = 0;
  for (auto op : this->GetAllOps()) {
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    if (!cop)
      continue;
    float fop = 0;
    for (auto b : cop->body) {
      fop += utils::GetFloatOps(b);
    }
    for (auto s : op.output(0)->shape) {
      const IntImmNode *as_int = s.as<IntImmNode>();
      CHECK(as_int) << "Please use static shape, rather than " << s << ".\n";
      fop = fop * as_int->value;
    }
    ret = ret + fop;
  }
  return FloatImm(tvm::runtime::DataType::Float(32), ret / 1e9);
}

Layer::Layer(std::string name, Array<te::Operation> ops,
             Array<te::Tensor> inputs, Array<te::Tensor> weights,
             Array<PrimExpr> const_scalars, Array<te::Tensor> const_tensors) {
  auto node = make_object<LayerNode>();
  node->name = name;
  node->ops = ops;
  node->inputs = inputs;
  node->weights = weights;
  node->const_scalars = const_scalars;
  node->const_tensors = const_tensors;
  // TODO: remove these constraints
  CHECK(const_scalars.size() == 0U)
      << "Currently please do not use const scalars.\n";
  CHECK(const_tensors.size() == 0U)
      << "Currently please do not use const tensors.\n";
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

  self->output_layer_tensors_.clear();
  int num_out = 0;
  for (te::Operation op : self->ops) {
    helper(op);
    CHECK(new_ops.count(op)) << "Missing op " << op << ".\n";
    LayerTensor tmp =
        LayerTensor(self->name, self, new_ops.at(op).output(0), num_out++);
    self->output_layer_tensors_.push_back(tmp);
  }

  return self->output_layer_tensors_;
}

// Block::Block(std::string name, Array<LayerTensor> out_tensors) {
//   auto node = make_object<BlockNode>();
//   node->name = name;
//   node->out_tensors = out_tensors;
//   data_ = node;
// }

Array<Layer> GraphNode::GetAllLayers() const {
  std::unordered_set<Layer> visit;
  std::vector<Layer> ret;

  std::function<void(Layer layer)> helper;
  helper = [&](Layer layer) {
    if (!layer.defined() || visit.count(layer))
      return;
    visit.insert(layer);
    for (auto inp : layer->input_layer_tensors_) {
      helper(inp->layer);
    }
    ret.push_back(layer);
  };

  for (auto lt : graph_outputs) {
    helper(lt->layer);
  }

  // left to right: inputs to outputs
  // std::reverse(ret.begin(), ret.end());
  return Array<Layer>(ret);
}

Array<Layer> FindConvexSet(Array<Layer> source, Array<Layer> sink) {
  std::unordered_set<Layer> visit;
  std::vector<Layer> ret;
  std::unordered_set<Layer> source_set;
  for (auto l : source) {
    source_set.insert(l);
  }

  std::function<bool(Layer layer)> helper;
  helper = [&](Layer layer) {
    if (!layer.defined() || visit.count(layer))
      return false;
    visit.insert(layer);
    bool reachable = false;
    if (!source_set.count(layer)) {
      for (auto inp : layer->input_layer_tensors_) {
        bool reach = helper(inp->layer);
        reachable = reachable || reach;
      }
    } else {
      reachable = true;
    }

    if (reachable) {
      ret.push_back(layer);
    }
    return reachable;
  };

  for (auto l : sink) {
    helper(l);
  }

  // left to right: inputs to outputs
  // std::reverse(ret.begin(), ret.end());
  return Array<Layer>(ret);
}

Graph::Graph(std::string name, Array<LayerTensor> graph_inputs,
             Array<LayerTensor> graph_outputs) {
  auto node = make_object<GraphNode>();
  node->name = name;
  node->graph_inputs = graph_inputs;
  node->graph_outputs = graph_outputs;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerTensor")
    .set_body_typed([](std::string name, Layer layer, te::Tensor tensor,
                       int value_idx) {
      return LayerTensor(name, layer, tensor, value_idx);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerTensorHash")
    .set_body_typed([](LayerTensor tensor) -> int64_t {
      return static_cast<int64_t>(std::hash<LayerTensor>()(tensor));
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerTensorEqual")
    .set_body_method(&LayerTensor::operator==);

TVM_REGISTER_GLOBAL("ditto.auto_compute.Layer")
    .set_body_typed([](std::string name, Array<te::Operation> ops,
                       Array<te::Tensor> inputs, Array<te::Tensor> weights,
                       Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerGetAllOps")
    .set_body_typed([](Layer layer) { return layer->GetAllOps(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerGetFingerprint")
    .set_body_typed([](Layer layer) { return layer->GetFingerprint(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerGetDataTransferAmount")
    .set_body_typed([](Layer layer) { return layer->GetDataTransferAmount(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerGetGFLOPS")
    .set_body_typed([](Layer layer) { return layer->GetGFLOPS(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.MakeLayer")
    .set_body_typed([](std::string name, Array<te::Operation> ops,
                       Array<te::Tensor> inputs, Array<te::Tensor> weights,
                       Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.LayerHash")
    .set_body_typed([](Layer layer) -> int64_t {
      return static_cast<int64_t>(std::hash<Layer>()(layer));
    });

// TVM_REGISTER_GLOBAL("ditto.auto_compute.Block")
//     .set_body_typed([](std::string name, Array<LayerTensor> out_tensors) {
//       return Block(name, out_tensors);
//     });

TVM_REGISTER_GLOBAL("ditto.auto_compute.Graph")
    .set_body_typed([](std::string name, Array<LayerTensor> graph_inputs,
                       Array<LayerTensor> graph_outputs) {
      return Graph(name, graph_inputs, graph_outputs);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.GraphGetAllLayers")
    .set_body_typed([](Graph graph) { return graph->GetAllLayers(); });

TVM_REGISTER_GLOBAL("ditto.auto_compute.MakeGraph")
    .set_body_typed([](std::string name, Array<LayerTensor> graph_inputs,
                       Array<LayerTensor> graph_outputs) {
      return Graph(name, graph_inputs, graph_outputs);
    });

TVM_REGISTER_GLOBAL("ditto.auto_compute.ProduceOutputs")
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

} // namespace auto_compute

} // namespace ditto