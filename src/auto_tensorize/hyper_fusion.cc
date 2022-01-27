#include <auto_tensorize/hyper_fusion.h>

namespace ditto {

namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(FusionChoiceNode);
TVM_REGISTER_NODE_TYPE(MatchInfoNode);
TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);
TVM_REGISTER_NODE_TYPE(CUDATensorizeContextNode);
TVM_REGISTER_NODE_TYPE(CUDATensorizeParamNode);

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

bool CUDATensorizeContextNode::HasEpilogue() {
  return (state->epilogue.size() > 0U);
}

te::Operation CUDATensorizeContextNode::EpilogueRootOp() {
  CHECK(this->HasEpilogue());
  return this->state->epilogue[(int)this->state->epilogue.size() - 1];
}

Array<te::Operation> CUDATensorizeContextNode::EpilogueNonRootOps() {
  Array<te::Operation> ret;
  for (int i = 0; i < (int)this->state->epilogue.size() - 1; ++i) {
    ret.push_back(this->state->epilogue[i]);
  }
  return ret;
}

Array<tir::IterVar> CUDATensorizeContextNode::Split(te::Schedule sch,
                                                    te::Operation op,
                                                    tir::IterVar iv,
                                                    Array<PrimExpr> factors) {
  std::vector<tir::IterVar> ret;
  int nparts = (int)factors.size();
  CHECK(nparts > 0);
  for (int i = nparts - 1; i > 0; --i) {
    tir::IterVar outer, inner;
    sch[op].split(iv, factors[i], &outer, &inner);
    iv = outer;
    ret.push_back(inner);
  }
  ret.push_back(iv);
  std::reverse(ret.begin(), ret.end());
  return Array<tir::IterVar>(ret);
}

tir::IterVar CUDATensorizeContextNode::FuseAll(te::Schedule sch,
                                               te::Operation op) {
  te::Operation sop = sch[op]->op;
  const te::ComputeOpNode *cop = sop.as<te::ComputeOpNode>();
  Array<tir::IterVar> axis = cop->axis;
  tir::IterVar fused;
  sch[op].fuse(axis, &fused);
  return fused;
}

Array<tir::IterVar>
CUDATensorizeContextNode::FuseAllAndSplit(te::Schedule sch, te::Operation op,
                                          Array<PrimExpr> factors) {
  tir::IterVar fused = this->FuseAll(sch, op);
  Array<tir::IterVar> tiled = this->Split(sch, op, fused, factors);
  return tiled;
}

void CUDATensorizeContextNode::Inline(te::Schedule sch, te::Operation op) {
  sch[op].compute_inline();
}

bool CUDATensorizeContextNode::CanInline(te::Operation op) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->reduce_axis.size() > 0U) {
    return false;
  }
  for (auto out : this->layer->ops) {
    if (out == op) {
      // is output op
      return false;
    }
  }
  return true;
}

std::pair<std::vector<int>, std::vector<int>>
CUDATensorizeContextNode::SecondOpOuterInnerSpatialAxis() {
  std::vector<int> outer_index;
  std::vector<int> inner_index;
  std::unordered_map<tir::IterVar, int> spatial_axis2index;
  std::unordered_set<int> outer_set;
  const te::ComputeOpNode *second_cop =
      this->state->second_op.as<te::ComputeOpNode>();
  CHECK(second_cop);
  int num_spatial_axis = (int)second_cop->axis.size();
  for (int i = 0; i < num_spatial_axis; ++i) {
    spatial_axis2index[second_cop->axis[i]] = i;
  }
  for (auto iv : this->state->fused_spatial_outer_iters) {
    CHECK(spatial_axis2index.count(iv));
    int idx = spatial_axis2index.at(iv);
    outer_index.push_back(idx);
    outer_set.insert(idx);
  }
  for (int i = 0; i < num_spatial_axis; ++i) {
    if (!outer_set.count(i)) {
      inner_index.push_back(i);
    }
  }
  return std::make_pair(outer_index, inner_index);
}

std::pair<std::vector<int>, std::vector<int>>
CUDATensorizeContextNode::SecondOpOuterInnerReduceAxis() {
  std::vector<int> outer_index;
  std::vector<int> inner_index;
  std::unordered_map<tir::IterVar, int> reduce_axis2index;
  std::unordered_set<int> outer_set;
  const te::ComputeOpNode *second_cop =
      this->state->second_op.as<te::ComputeOpNode>();
  CHECK(second_cop);
  int num_reduce_axis = (int)second_cop->reduce_axis.size();
  for (int i = 0; i < num_reduce_axis; ++i) {
    reduce_axis2index[second_cop->reduce_axis[i]] = i;
  }
  for (auto iv : this->state->fused_reduce_outer_iters) {
    CHECK(reduce_axis2index.count(iv));
    int idx = reduce_axis2index.at(iv);
    outer_index.push_back(idx);
    outer_set.insert(idx);
  }
  for (int i = 0; i < num_reduce_axis; ++i) {
    if (!outer_set.count(i)) {
      inner_index.push_back(i);
    }
  }
  return std::make_pair(outer_index, inner_index);
}

std::vector<int> CUDATensorizeContextNode::SecondOpTensorizeSpatialAxis() {
  CHECK(this->state->tensorize_iters.count(this->state->second_op));
  Array<tir::IterVar> iters =
      this->state->tensorize_iters.at(this->state->second_op);
  std::unordered_map<tir::IterVar, int> spatial_axis2index;
  const te::ComputeOpNode *second_cop =
      this->state->second_op.as<te::ComputeOpNode>();
  CHECK(second_cop);
  int num_spatial_axis = (int)second_cop->axis.size();
  for (int i = 0; i < num_spatial_axis; ++i) {
    spatial_axis2index[second_cop->axis[i]] = i;
  }
  std::vector<int> ret;
  for (auto iv : iters) {
    if (spatial_axis2index.count(iv)) {
      ret.push_back(spatial_axis2index.at(iv));
    }
  }
  return ret;
}

std::vector<int> CUDATensorizeContextNode::SecondOpTensorizeReduceAxis() {
  CHECK(this->state->tensorize_iters.count(this->state->second_op));
  Array<tir::IterVar> iters =
      this->state->tensorize_iters.at(this->state->second_op);
  std::unordered_map<tir::IterVar, int> reduce_axis2index;
  const te::ComputeOpNode *second_cop =
      this->state->second_op.as<te::ComputeOpNode>();
  CHECK(second_cop);
  int num_reduce_axis = (int)second_cop->reduce_axis.size();
  for (int i = 0; i < num_reduce_axis; ++i) {
    reduce_axis2index[second_cop->reduce_axis[i]] = i;
  }
  std::vector<int> ret;
  for (auto iv : iters) {
    if (reduce_axis2index.count(iv)) {
      ret.push_back(reduce_axis2index.at(iv));
    }
  }
  return ret;
}

bool CUDATensorizeContextNode::ValidTensorizeFusion(
    const std::vector<int> &inner_index,
    const std::vector<int> &tensorize_index) {
  int len1 = (int)inner_index.size();
  int len2 = (int)tensorize_index.size();
  if (len2 > len1) {
    return false;
  }
  // the tensorized iters should be innermost loops
  // e.g., inner: [i1, i2, i3, i4], tensorize: [i3, i4]
  for (int i = 0; i < len2; ++i) {
    if (inner_index[i + len1 - len2] != tensorize_index[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> CUDATensorizeContextNode::GetSpatialExtentsByIndex(
    const te::Operation &op, const std::vector<int> &index) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  int num_axis = (int)cop->axis.size();
  std::vector<int> ret;
  for (auto ind : index) {
    CHECK(ind < num_axis);
    tir::IterVar iv = cop->axis[ind];
    PrimExpr ext = iv->dom->extent;
    const IntImmNode *as_int = ext.as<IntImmNode>();
    CHECK(as_int) << "Currently only static shape is supported.\n";
    ret.push_back(as_int->value);
  }
  return ret;
}

std::vector<int> CUDATensorizeContextNode::GetReduceExtentsByIndex(
    const te::Operation &op, const std::vector<int> &index) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  int num_axis = (int)cop->reduce_axis.size();
  std::vector<int> ret;
  for (auto ind : index) {
    CHECK(ind < num_axis);
    tir::IterVar iv = cop->reduce_axis[ind];
    PrimExpr ext = iv->dom->extent;
    const IntImmNode *as_int = ext.as<IntImmNode>();
    CHECK(as_int) << "Currently only static shape is supported.\n";
    ret.push_back(as_int->value);
  }
  return ret;
}

CUDATensorizeContext::CUDATensorizeContext(Layer layer,
                                           TensorizeHyperFusionState state,
                                           hardware::HardwareParam cuda_param) {
  auto node = make_object<CUDATensorizeContextNode>();
  node->layer = layer;
  node->state = state;
  node->cuda_param = cuda_param;
  data_ = node;
}

CUDATensorizeParam::CUDATensorizeParam(int warp_size, int ty_size, int tz_size,
                                       int input_vector_len, int serial_y,
                                       int serial_z, int block_rx, int block_ry,
                                       int block_rz, int warp_rx, int warp_ry,
                                       int warp_rz) {
  auto node = make_object<CUDATensorizeParamNode>();
  node->warp_size = warp_size;
  node->ty_size = ty_size;
  node->tz_size = tz_size;
  node->input_vector_len = input_vector_len;
  node->serial_y = serial_y;
  node->serial_z = serial_z;
  node->block_rx = block_rx;
  node->block_ry = block_ry;
  node->block_rz = block_rz;
  node->warp_rx = warp_rx;
  node->warp_ry = warp_ry;
  node->warp_rz = warp_rz;
  data_ = node;
}

void ScheduleEpilogue(te::Schedule sch, CUDATensorizeContext ctx,
                      CUDATensorizeParam tensorize_param) {
  te::Operation cur_op;
  if (ctx->HasEpilogue()) {
    cur_op = ctx->EpilogueRootOp();
    Array<tir::IterVar> tiled =
        ctx->FuseAllAndSplit(sch, cur_op, {-1, tensorize_param->warp_size});
    sch[cur_op].bind(tiled[0], te::thread_axis(Range(), "blockIdx.x"));
    sch[cur_op].bind(tiled[1], te::thread_axis(Range(), "threadIdx.x"));
    Array<te::Operation> remain_ops = ctx->EpilogueNonRootOps();
    // the remaining non-root ops should be inlined
    for (auto op : remain_ops) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
    }
  }
}

void ScheduleSecondOpParallelism(te::Schedule sch, CUDATensorizeContext ctx,
                                 CUDATensorizeParam tensorize_param) {
  te::Operation cur_op;
  CHECK(ctx->state->second_op->num_outputs() == 1U);
  te::Tensor second_tensor = ctx->state->second_op.output(0);
  CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->second_op));
  PackedIntrinsic pintrin =
      ctx->state->tensorize_intrinsics.at(ctx->state->second_op);
  te::Tensor second_frag =
      sch.cache_write(second_tensor, pintrin->compute_scope);
  // find the spatial outer axis
  std::vector<int> outer_index, inner_index, tensorize_index;
  std::tie(outer_index, inner_index) = ctx->SecondOpOuterInnerSpatialAxis();
  tensorize_index = ctx->SecondOpTensorizeSpatialAxis();
  CHECK(ctx->ValidTensorizeFusion(inner_index, tensorize_index))
      << "The fusion and tensorize decisions are not valid.\n";
  inner_index.erase(inner_index.begin() +
                        (inner_index.size() - tensorize_index.size()),
                    inner_index.end());
  // outer_index should be split into 3 parts
  // inner_index should be split into 2 parts
  // tensorize_index shouldn't be split
  std::vector<int> outer_extents, inner_extents;
  outer_extents =
      ctx->GetSpatialExtentsByIndex(ctx->state->second_op, outer_index);
  inner_extents =
      ctx->GetSpatialExtentsByIndex(ctx->state->second_op, inner_index);
  /* The following conditions are considered:
   * 1. Two inner axis large enough for y and z dim;
   * 2. One inner axis large enough for y or z dim;
   * 3. No inner axis is large enough
   */
  int z_dim_index{-1}, y_dim_index{-1};
  for (int i = 0; i < (int)inner_extents.size(); ++i) {
    if ((inner_extents[i] >= tensorize_param->tz_size) && z_dim_index < 0) {
      z_dim_index = i;
    } else if ((inner_extents[i] >= tensorize_param->ty_size) &&
               y_dim_index < 0) {
      y_dim_index = i;
    }
  }
  /* The following conditions are considered:
   * 1. Two outer axis large enough for y and z dim;
   * 2. One outer axis large enough for y or z dim;
   * 3. No outer axis is large enough
   */
  int z_block_index{-1}, y_block_index{-1};
  for (int i = 0; i < (int)outer_extents.size(); ++i) {
    if ((outer_extents[i] >= tensorize_param->tz_size) && z_block_index < 0) {
      z_block_index = i;
    } else if ((outer_extents[i] >= tensorize_param->ty_size) &&
               y_block_index < 0) {
      y_block_index = i;
    }
  }
  // use these vectors to store intermediate axis
  std::vector<tir::IterVar> outer_outer(outer_index.size(), tir::IterVar()),
      outer_inner(outer_index.size(), tir::IterVar()),
      outer_inner_inner(outer_index.size(), tir::IterVar()),
      inner_outer(inner_index.size(), tir::IterVar()),
      inner_inner(inner_index.size(), tir::IterVar()), tensor_iters;
  cur_op = sch[second_tensor]->op;
  const te::ComputeOpNode *cur_cop = cur_op.as<te::ComputeOpNode>();
  CHECK(cur_cop);
  // iters for tensorize
  for (auto ind : tensorize_index) {
    CHECK((int)cur_cop->axis.size() > ind);
    tensor_iters.push_back(cur_cop->axis[ind]);
  }
  bool ever_bind{false};
  // first, bind thread y
  if (y_dim_index >= 0) {
    // use inner axis for thread y
    CHECK((int)inner_index.size() > y_dim_index);
    int ind = inner_index[y_dim_index];
    CHECK((int)cur_cop->axis.size() > ind);
    tir::IterVar axis = cur_cop->axis[ind];
    tir::IterVar outer, inner;
    sch[second_tensor].split_by_nparts(axis, tensorize_param->ty_size, &outer,
                                       &inner);
    sch[second_tensor].bind(outer, te::thread_axis(Range(), "threadIdx.y"));
    ever_bind = true; // form a valid GPU kernel
    inner_outer[y_dim_index] = outer;
    inner_inner[y_dim_index] = inner;
  }
  if (y_block_index >= 0) {
    CHECK((int)outer_index.size() > y_block_index);
    int ind = outer_index[y_block_index];
    CHECK((int)cur_cop->axis.size() > ind);
    tir::IterVar axis = cur_cop->axis[ind];
    if (y_dim_index < 0) {
      // use outer axis for block x and thread y
      Array<tir::IterVar> tiled =
          ctx->Split(sch, second_tensor->op, axis,
                     {-1, tensorize_param->ty_size, tensorize_param->serial_y});
      sch[second_tensor].bind(tiled[0], te::thread_axis(Range(), "blockIdx.x"));
      sch[second_tensor].bind(tiled[1],
                              te::thread_axis(Range(), "threadIdx.y"));
      ever_bind = true; // form a valid GPU kernel
      outer_outer[y_block_index] = tiled[0];
      outer_inner[y_block_index] = tiled[1];
      outer_inner_inner[y_block_index] = tiled[2];
    } else {
      sch[second_tensor].bind(axis, te::thread_axis(Range(), "blockIdx.x"));
    }
  }
  // then, bind thread z
  if (z_dim_index >= 0) {
    // use inner axis for thread z
    CHECK((int)inner_index.size() > z_dim_index);
    int ind = inner_index[z_dim_index];
    CHECK((int)cur_cop->axis.size() > ind);
    tir::IterVar axis = cur_cop->axis[ind];
    tir::IterVar outer, inner;
    sch[second_tensor].split_by_nparts(axis, tensorize_param->tz_size, &outer,
                                       &inner);
    sch[second_tensor].bind(outer, te::thread_axis(Range(), "threadIdx.z"));
    ever_bind = true; // form a valid GPU kernel
    inner_outer[z_dim_index] = outer;
    inner_inner[z_dim_index] = inner;
  }
  if (z_block_index >= 0) {
    CHECK((int)outer_index.size() > z_block_index);
    int ind = outer_index[z_block_index];
    CHECK((int)cur_cop->axis.size() > ind);
    tir::IterVar axis = cur_cop->axis[ind];
    if (z_dim_index < 0) {
      // use outer axis for block y and thread z
      Array<tir::IterVar> tiled =
          ctx->Split(sch, second_tensor->op, axis,
                     {-1, tensorize_param->tz_size, tensorize_param->serial_z});
      sch[second_tensor].bind(tiled[0], te::thread_axis(Range(), "blockIdx.y"));
      sch[second_tensor].bind(tiled[1],
                              te::thread_axis(Range(), "threadIdx.z"));
      ever_bind = true; // form a valid GPU kernel
      outer_outer[z_block_index] = tiled[0];
      outer_inner[z_block_index] = tiled[1];
      outer_inner_inner[z_block_index] = tiled[2];
    } else {
      sch[second_tensor].bind(axis, te::thread_axis(Range(), "blockIdx.y"));
    }
  }
  // finally, bind block z
  std::vector<tir::IterVar> bind_block_z;
  for (int i = 0; i < (int)outer_index.size(); ++i) {
    if ((i != y_block_index) && (i != z_block_index)) {
      // this outer axis is never bound
      bind_block_z.push_back(cur_cop->axis[outer_index[i]]);
    }
  }
  // collect the remaining inner axis
  for (int i = 0; i < (int)inner_index.size(); ++i) {
    if ((i != y_dim_index) && (i != z_dim_index)) {
      // this inner axis is never bound
      inner_outer[i] = cur_cop->axis[inner_index[i]];
    }
  }
  // Reorder all the axis
  Array<tir::IterVar> order;
  for (auto list : {bind_block_z, outer_outer, outer_inner, inner_outer,
                    outer_inner_inner, inner_inner, tensor_iters}) {
    for (auto iv : list) {
      if (iv.defined()) {
        order.push_back(iv);
      }
    }
  }
  sch[second_tensor].reorder(order);
  tir::IterVar fused_block_z;
  if (bind_block_z.size() > 0U) {
    sch[second_tensor].fuse(bind_block_z, &fused_block_z);
    sch[second_tensor].bind(fused_block_z,
                            te::thread_axis(Range(), "blockIdx.z"));
    ever_bind = true; // form a valid GPU kernel
  }
  CHECK(ever_bind) << "The scheduler can't bind any axis for GPU.\n";
  /*
   * Find postion to compute at
   */
  tir::IterVar pos;
  for (auto list : {inner_outer, outer_inner, outer_outer}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        pos = list[i];
        break;
      }
    }
    if (pos.defined()) {
      break;
    }
  }
  if (!pos.defined()) {
    pos = fused_block_z;
  }
  CHECK(pos.defined())
      << "Can't find second_frag compute_at position during scheduling.\n";
  /*
   * Store the context for following schedules
   */
  ctx->second_op_compute_pos = pos;
  ctx->second_frag = second_frag;
}

void ScheduleSecondOpLocality(te::Schedule sch, CUDATensorizeContext ctx,
                              CUDATensorizeParam tensorize_param) {
  te::Tensor second_frag = ctx->second_frag;
  CHECK(second_frag.defined() && sch[second_frag]->op.defined());
  const te::ComputeOpNode *frag_cop =
      sch[second_frag]->op.as<te::ComputeOpNode>();
  CHECK(frag_cop != nullptr)
      << second_frag << " is from " << sch[second_frag]->op << "\n";
  /*
   * Tile the reduce axis with largest extent
   */
  // find the spatial outer axis
  std::vector<int> outer_index, inner_index, tensorize_index;
  std::tie(outer_index, inner_index) = ctx->SecondOpOuterInnerSpatialAxis();
  tensorize_index = ctx->SecondOpTensorizeSpatialAxis();
}

te::Schedule TensorizeCUDA(Layer layer, TensorizeHyperFusionState state,
                           hardware::HardwareParam cuda_param,
                           CUDATensorizeParam tensorize_param) {
  te::Schedule sch = te::create_schedule(layer->ops);
  CUDATensorizeContext ctx = CUDATensorizeContext(layer, state, cuda_param);
  /* Schedule epilogue
   * Currently, we treat epilogue as a separate kernel
   */
  ScheduleEpilogue(sch, ctx, tensorize_param);
  /* Schedule second op tensor
   * The second tensor determines the overall parallelism
   */
  ScheduleSecondOpParallelism(sch, ctx, tensorize_param);
  /* Schedule second op fragment
   * The second frag determines part of the locality
   */
  ScheduleSecondOpLocality(sch, ctx, tensorize_param);
  return sch;
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

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CUDATensorizeContext")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                       hardware::HardwareParam cuda_param) {
      return CUDATensorizeContext(layer, state, cuda_param);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CUDATensorizeParam")
    .set_body_typed([](int warp_size, int ty_size, int tz_size,
                       int input_vector_len, int serial_y, int serial_z,
                       int block_rx, int block_ry, int block_rz, int warp_rx,
                       int warp_ry, int warp_rz) {
      return CUDATensorizeParam(warp_size, ty_size, tz_size, input_vector_len,
                                serial_y, serial_z, block_rx, block_ry,
                                block_rz, warp_rx, warp_ry, warp_rz);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeCUDA")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                       hardware::HardwareParam cuda_param,
                       CUDATensorizeParam tensorize_param) {
      return TensorizeCUDA(layer, state, cuda_param, tensorize_param);
    });

} // namespace auto_tensorize

} // namespace ditto