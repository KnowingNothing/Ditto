#include <auto_tensorize/hyper_fusion.h>

namespace ditto {

namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(FusionChoiceNode);
TVM_REGISTER_NODE_TYPE(MatchInfoNode);
TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);

FusionChoice::FusionChoice(te::Operation first_op, te::Operation second_op,
                           Array<tir::IterVar> ordered_iters, int attach_pos) {
  auto node = make_object<FusionChoiceNode>();
  node->first_op = first_op;
  node->second_op = second_op;
  node->ordered_iters = ordered_iters;
  node->attach_pos = attach_pos;
  data_ = node;
}

MatchInfo::MatchInfo(Array<tir::IterVar> axis, PackedIntrinsic intrin) {
  auto node = make_object<MatchInfoNode>();
  node->axis = axis;
  node->intrin = intrin;
  data_ = node;
}

TensorizeHyperFusionState::TensorizeHyperFusionState(
    Layer layer, FusionChoice fuse_choice,
    Map<te::Operation, MatchInfo> match_info) {
  auto node = make_object<TensorizeHyperFusionStateNode>();
  //   Array<Array<te::Operation>> first_op_prologue;
  //   Array<Array<te::Operation>> second_op_prologue;
  //   Array<te::Operation> inter_path;
  //   Array<te::Operation> epilogue;
  node->layer = layer;
  node->first_op = fuse_choice->first_op;
  node->second_op = fuse_choice->second_op;

  CHECK(fuse_choice->attach_pos < (int)fuse_choice->ordered_iters.size());
  for (int i = 0; i <= fuse_choice->attach_pos; ++i) {
    tir::IterVar iv = fuse_choice->ordered_iters[i];
    if (iv->iter_type == tir::IterVarType::kDataPar) {
      node->fused_spatial_outer_iters.push_back(iv);
    } else if (iv->iter_type == tir::IterVarType::kCommReduce) {
      node->fused_reduce_outer_iters.push_back(iv);
    } else {
      CHECK(false) << "Unexpected iter_type " << iv->iter_type << ".\n";
    }
  }

  /* Traverse the DAT to capture desired ops
   * We use some global states to aid this process
   */
  // global op flag
  bool first_op_found = false;
  bool second_op_found = false;
  // the ops visited till current op
  std::unordered_map<te::Operation, Array<te::Operation>> prefix_array;
  // return <first_op_met, second_op_met>
  std::function<std::pair<bool, bool>(te::Operation)> helper;
  helper = [&](te::Operation op) {
    // TODO: finish this function
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    if (!cop) {
      return std::make_pair(false, false);
    }
    CHECK(!prefix_array.count(op))
        << "The same op " << op << " is visited multiple times.\n"
        << "The given layer is not a tree structure.\n"
        << "Layer information:\n"
        << layer;
    bool met_first{false}, met_second{false};
    if (op == node->first_op) {
      for (auto inp : cop->InputTensors()) {
        bool inp_met_first{false}, inp_met_second{false};
        std::tie(inp_met_first, inp_met_second) = helper(inp->op);
        CHECK((!first_op_found) && (!second_op_found));
        CHECK((!inp_met_first) && (!inp_met_second));
        if (prefix_array.count(inp->op)) {
          node->first_op_prologue.push_back(prefix_array.at(inp->op));
        }
      }
      // globally update the first_op_found
      first_op_found = true;
      met_first = true;
    } else if (op == node->second_op) {
      bool first_op_in_producer{false};
      for (auto inp : cop->InputTensors()) {
        bool inp_met_first{false}, inp_met_second{false};
        std::tie(inp_met_first, inp_met_second) = helper(inp->op);
        CHECK(!inp_met_second);
        CHECK(first_op_found && (!second_op_found));
        // update to see if first_op is in producers
        first_op_in_producer |= inp_met_first;
        if (prefix_array.count(inp->op)) {
          if (inp_met_first) {
            // belong to path
            CHECK(node->inter_path.size() == 0U)
                << "Multiple paths from first op to second op.\n";
            node->inter_path = prefix_array.at(inp->op);
          } else {
            // prologue
            node->second_op_prologue.push_back(prefix_array.at(inp->op));
          }
        }
      }
      CHECK(first_op_in_producer)
          << "First op is not the producer of second op.\n";
      // globally update the second_op_found
      second_op_found = true;
      met_second = true;
    } else {
      for (auto inp : cop->InputTensors()) {
        bool inp_met_first{false}, inp_met_second{false};
        std::tie(inp_met_first, inp_met_second) = helper(inp->op);
        met_first |= inp_met_first;
        met_second |= inp_met_second;
        if (prefix_array.count(inp->op)) {
          for (auto v : prefix_array.at(inp->op)) {
            prefix_array[op].push_back(v);
          }
        }
      }
      prefix_array[op].push_back(op);
    }

    return std::make_pair(met_first, met_second);
  };

  CHECK(layer->ops.size() == 1U);
  helper(layer->ops[0]);
  if (prefix_array.count(layer->ops[0])) {
    node->epilogue = prefix_array.at(layer->ops[0]);
  }

  CHECK(match_info.size() == 2U);
  CHECK(match_info.count(node->first_op) && match_info.count(node->second_op));
  node->tensorize_iters.Set(node->first_op,
                            match_info.at(node->first_op)->axis);
  node->tensorize_intrinsics.Set(node->first_op,
                                 match_info.at(node->first_op)->intrin);
  node->tensorize_iters.Set(node->second_op,
                            match_info.at(node->second_op)->axis);
  node->tensorize_intrinsics.Set(node->second_op,
                                 match_info.at(node->second_op)->intrin);
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.FusionChoice")
    .set_body_typed([](te::Operation first_op, te::Operation second_op,
                       Array<tir::IterVar> ordered_iters, int attach_pos) {
      return FusionChoice(first_op, second_op, ordered_iters, attach_pos);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.MatchInfo")
    .set_body_typed([](Array<tir::IterVar> axis, PackedIntrinsic intrin) {
      return MatchInfo(axis, intrin);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeHyperFusionState")
    .set_body_typed([](Layer layer, FusionChoice fuse_choice,
                       Map<te::Operation, MatchInfo> match_info) {
      return TensorizeHyperFusionState(layer, fuse_choice, match_info);
    });

} // namespace auto_tensorize

} // namespace ditto