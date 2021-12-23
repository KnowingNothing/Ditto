#include <auto_tensorize/pattern.h>
#include <auto_tensorize/state.h>
#include <utils/iter_domain.h>

namespace ditto {

namespace auto_tensorize {

AccessFunction::AccessFunction(te::Operation op,
                               Array<Array<PrimExpr>> access_indices) {
  auto node = make_object<AccessFunctionNode>();
  node->op = op;
  node->access_indices = access_indices;
  data_ = node;
}

Array<tir::IterVar> OpHyperStateNode::SpatialIters() {
  return op.as<te::ComputeOpNode>()->axis;
}

Array<tir::IterVar> OpHyperStateNode::ReduceIters() {
  return op.as<te::ComputeOpNode>()->reduce_axis;
}

Array<AccessFunction> OpHyperStateNode::ReadAccessFunctions() {
  Array<AccessFunction> ret;
  for (auto inp : op->InputTensors()) {
    Array<Array<PrimExpr>> access_indices =
        utils::GetAccessIndices(op, inp->op);
    AccessFunction func(inp->op, access_indices);
    ret.push_back(func);
  }
  return ret;
}

AccessFunction OpHyperStateNode::WriteAccessFunctions() {
  Array<Array<PrimExpr>> access_indices;
  Array<PrimExpr> indices;
  for (auto iv : SpatialIters()) {
    indices.push_back(iv->var);
  }
  access_indices.push_back(indices);
  return AccessFunction(op, access_indices);
}

OpHyperState::OpHyperState(te::Operation op) {
  auto node = make_object<OpHyperStateNode>();
  CHECK(op.as<te::ComputeOpNode>() != nullptr);
  node->op = op;
  node->pattern = GetOpPattern(op);
  data_ = node;
}

bool SerialFusionStateNode::IsLinearTopo() {
  for (auto kv : layer_state->feed_graph) {
    te::Operation key = kv.first;
    if (op_hyper_states.count(key)) {
      // this is a compute op
      if (kv.second.size() > 1U) {
        /* a compute op has more than one
          consumer, which is not linear
          e.g.,
                     o
                   /   \
                  o     o
        */
        return false;
      }
    }
  }
  return true;
}

SerialFusionState::SerialFusionState(Layer layer) {
  auto node = make_object<SerialFusionStateNode>();
  node->layer_state = LayerState(layer);
  for (auto op : layer->GetAllOps()) {
    // we only consider compute op
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    if (cop != nullptr) {
      node->op_hyper_states[op] = OpHyperState(op);
    }
  }
  data_ = node;
}

} // namespace auto_tensorize

} // namespace ditto