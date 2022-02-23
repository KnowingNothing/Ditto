#include <auto_tensorize/pattern.h>
#include <auto_tensorize/state.h>
#include <utils/iter_domain.h>

namespace ditto {

namespace auto_tensorize {

Array<IterVar> OpHyperStateNode::getAllIters() {
  Array<IterVar> ret;

  int index = 0;
  for (auto iv : op.as<te::ComputeOpNode>()->axis) {
    if (IterMap.count(iv) == 0) {
      const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
      IterMap[iv] =
          IterVar(index, as_int->value, IV_Type::SPATIAL,
                  tir::Var(op->name + "." + iv->var->name_hint), iv->var);
    }
    ret.push_back(IterMap.at(iv));
    index += 1;
  }
  for (auto iv : op.as<te::ComputeOpNode>()->reduce_axis) {
    if (IterMap.count(iv) == 0) {
      const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
      IterMap[iv] =
          IterVar(index, as_int->value, IV_Type::REDUCE,
                  tir::Var(op->name + "." + iv->var->name_hint), iv->var);
    }
    ret.push_back(IterMap.at(iv));
    index += 1;
  }
  return ret;
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

OpHyperState::OpHyperState(te::Operation op, size_t index) {
  auto node = make_object<OpHyperStateNode>();
  CHECK(op.as<te::ComputeOpNode>() != nullptr);
  node->op = op;
  node->pattern = GetOpPattern(op);
  node->index = index;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SerialFusionStateNode);

std::pair<OpHyperState, OpHyperState> SerialFusionStateNode::getCubicOpPair() {
  std::pair<OpHyperState, OpHyperState> ret;
  int cnt = 0;
  int idx1, idx2;
  for (std::pair<te::Operation, OpHyperState> s : op_hyper_states) {
    if (s.second->pattern == OpPattern::PATTERN_CUBIC) {
      if (cnt == 0) {
        ret.first = s.second;
        idx1 = s.second->index;
      }
      if (cnt == 1) {
        ret.second = s.second;
        idx2 = s.second->index;
      }
      cnt++;
    }
  }
  CHECK(cnt == 2) << "we only deal with 2 CUBIC ops, but has " << cnt;
  if (idx1 > idx2)
    std::swap(ret.first, ret.second);
  return ret;
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

int SerialFusionStateNode::CountOp(OpPattern pattern) {
  int counter = 0;
  for (auto op : this->ops) {
    if (this->op_hyper_states[op]->pattern == pattern) {
      counter += 1;
    }
  }
  return counter;
}

SerialFusionState::SerialFusionState(Layer layer) {
  auto node = make_object<SerialFusionStateNode>();
  node->layer_state = LayerState(layer);
  size_t idx = 0;
  for (auto op : layer->GetAllOps()) {
    // we only consider compute op
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    if (cop != nullptr) {
      node->ops.push_back(op);
      node->op_hyper_states[op] = OpHyperState(op, idx);
    }
    idx++;
  }
  data_ = node;
}
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.build_serial_fusion_state")
    .set_body_typed(buildSerialFusionState);

} // namespace auto_tensorize

} // namespace ditto