#include <auto_tensorize/pattern.h>
#include <auto_tensorize/state.h>
#include <utils/iter_domain.h>
#include <auto_tensorize/analysis.h>

namespace ditto {

namespace auto_tensorize {
Array<tir::IterVar> OpHyperStateNode::tensorizeAxes;
Map<tir::IterVar, tir::IterVar> OpHyperStateNode::tensorizeMap; 
Array<IterVar> OpHyperStateNode::getAllIters() {
  Array<IterVar> ret;

  int index = 0;
  CHECK(op->InputTensors().size() == 2);
  te::Operation op1 = op->InputTensors()[0]->op,
                op2 = op->InputTensors()[1]->op;
  Array<tir::Var> op1Vars = utils::GetAccessVars(op, op1);
  Array<tir::Var> op2Vars = utils::GetAccessVars(op, op2);
  std::function<bool(tir::IterVar)> isTensroize = [&](tir::IterVar iv) {
    for (auto iv_ : tensorizeAxes) {
      if (iv.same_as(iv_))
        return true;
    }
    return false;
  };
  auto in = [](tir::Var var_, Array<tir::Var> varList) {
    for (auto var : varList)
      if (var.same_as(var_))
        return true;
    return false;
  };
  for (auto iv : op.as<te::ComputeOpNode>()->axis) {
    if (IterMap.count(iv) == 0) {
      const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
      IV_Type iv_type = IV_Type::REDUCE;
      bool in_op1 = in(iv->var, op1Vars);
      bool in_op2 = in(iv->var, op2Vars);
      if (isTensroize(iv))
        iv_type = IV_Type::TENSORIZE;
      else if (in_op1 && in_op2)
        iv_type = IV_Type::BATCH;
      else if (in_op1)
        iv_type = IV_Type::FIRSTSPATIAL;
      else if (in_op2)
        iv_type = IV_Type::SECONDSPATIAL;
      CHECK(iv_type != IV_Type::REDUCE)
          << "neither first spatial nor second spatial";
      int tensorize_ext = 1;
      if (tensorizeMap.count(iv))
        tensorize_ext = tensorizeMap[iv]->dom->extent.as<tir::IntImmNode>()->value;
      IterMap[iv] =
          IterVar(index, as_int->value, iv_type,
                  tir::Var(op->name + "." + iv->var->name_hint), iv->var, tensorize_ext);
    }
    ret.push_back(IterMap.at(iv));
    index += 1;
  }
  for (auto iv : op.as<te::ComputeOpNode>()->reduce_axis) {
    if (IterMap.count(iv) == 0) {
      const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
      int tensorize_ext = 1;
      if (tensorizeMap.count(iv))
        tensorize_ext = tensorizeMap[iv]->dom->extent.as<tir::IntImmNode>()->value;
      IV_Type iv_type = isTensroize(iv) ? IV_Type::TENSORIZE : IV_Type::REDUCE;
      IterMap[iv] =
          IterVar(index, as_int->value, iv_type,
                  tir::Var(op->name + "." + iv->var->name_hint), iv->var, tensorize_ext);
    }
    ret.push_back(IterMap.at(iv));
    index += 1;
  }
  return ret;
}

Array<tir::Var> OpHyperStateNode::getAllVars() {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  Array<tir::Var> vars;
  for (auto iv : cop->axis)
    vars.push_back(iv->var);
  for (auto iv : cop->reduce_axis)
    vars.push_back(iv->var);
  return vars;
}

Array<tir::IterVar> OpHyperStateNode::SpatialIters() {
  return op.as<te::ComputeOpNode>()->axis;
}

Array<tir::IterVar> OpHyperStateNode::ReduceIters() {
  return op.as<te::ComputeOpNode>()->reduce_axis;
}
Array<Array<PrimExpr>> OpHyperStateNode::getAccessIndices(te::Operation op, te::Operation inop){
  Array<Array<PrimExpr>> access_indices = utils::GetAccessIndices(op, inop);
  op = inop;
  while (op.as<te::ComputeOpNode>() && op->InputTensors().size() == 1){
    Map<tir::Var, PrimExpr> map;
    const te::ComputeOpNode * cop = op.as<te::ComputeOpNode>();
    CHECK(cop->reduce_axis.size() == 0);
    CHECK(access_indices.size() == 1);
    CHECK(cop->axis.size() == access_indices[0].size()) << cop->axis << " " << access_indices;
    size_t i = 0;
    for (auto iv: cop->axis){
      map.Set(iv->var, access_indices[0][i++]);
    }
    Array<Array<PrimExpr>> tmp = utils::GetAccessIndices(op, op->InputTensors()[0]->op);
    if (tmp.size() > 1)
      break;
    Array<Array<PrimExpr>> new_acecss_indices;
    for (auto exprs: tmp){
      Array<PrimExpr> singleAccessIndices;
      for (auto expr: exprs){
        singleAccessIndices.push_back(utils::ReplaceVars(expr, map));
      }
      new_acecss_indices.push_back(singleAccessIndices);
    }
    access_indices = new_acecss_indices;
    op = op->InputTensors()[0]->op;
  }
  return access_indices;
}
Array<AccessFunction> OpHyperStateNode::ReadAccessFunctions() {
  Array<AccessFunction> ret;
  Array<tir::Var> allVars;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  for (auto iv : cop->axis)
    allVars.push_back(iv->var);
  for (auto iv : cop->reduce_axis)
    allVars.push_back(iv->var);
  for (auto inp : op->InputTensors()) {
    Array<tir::Var> opVars = utils::GetAccessVars(op, inp->op);
    Array<tir::Var> absentVars;
    Array<tir::Var> presentVars;
    for (auto var : allVars) {
      bool isAbsent = true;
      for (auto var_ : opVars) {
        if (var.same_as(var_)) {
          isAbsent = false;
          break;
        }
      }
      if (isAbsent)
        absentVars.push_back(var);
      else
        presentVars.push_back(var);
    }
    Array<Array<PrimExpr>> access_indices_old =
        utils::GetAccessIndices(op, inp->op);
    Array<Array<PrimExpr>> access_indices = 
        getAccessIndices(op, inp->op);
    AccessFunction func(inp->op, access_indices, absentVars, presentVars);
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
  Array<tir::Var> absentVars, presentVars;
  for (auto iv : op.as<te::ComputeOpNode>()->reduce_axis)
    absentVars.push_back(iv->var);
  for (auto iv : op.as<te::ComputeOpNode>()->axis)
    presentVars.push_back(iv->var);
  return AccessFunction(op, access_indices, absentVars, presentVars);
}

Map<tir::Var, IntImm> OpHyperStateNode::getBounds() {
  Map<tir::Var, IntImm> bounds;
  for (auto iv : getAllIters()) {
    bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
  }
  return bounds;
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

TVM_REGISTER_NODE_TYPE(OpHyperStateNode);

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
  OpHyperState op1, op2;
  std::tie(op1, op2) = node->getCubicOpPair();
  node->first_op = op1->op;
  node->second_op = op2->op;
  data_ = node;
}
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.build_serial_fusion_state")
    .set_body_typed(buildSerialFusionState);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.registerTensorizeAxes")
    .set_body_method<SerialFusionState>(&SerialFusionStateNode::registerTensorizeAxes);

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.registerTensorizeMap")
  .set_body_method<SerialFusionState>(&SerialFusionStateNode::registerTensorizeMap);
} // namespace auto_tensorize

} // namespace ditto