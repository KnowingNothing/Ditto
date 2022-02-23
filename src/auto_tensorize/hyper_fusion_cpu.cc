#include <auto_tensorize/analysis.h>
#include <auto_tensorize/hyper_fusion.h>
#include <tvm/te/schedule_pass.h>

namespace ditto {

namespace auto_tensorize {
    
TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);
TVM_REGISTER_NODE_TYPE(CPUTensorizeContextNode);
TVM_REGISTER_NODE_TYPE(CPUTensorizeParamNode);


bool CPUTensorizeContextNode::HasEpilogue() {
  return (state->epilogue.size() > 0U);
}

te::Operation CPUTensorizeContextNode::EpilogueRootOp() {
  CHECK(this->HasEpilogue());
  return this->state->epilogue[(int)this->state->epilogue.size() - 1];
}

Array<te::Operation> CPUTensorizeContextNode::EpilogueNonRootOps() {
  Array<te::Operation> ret;
  for (int i = 0; i < (int)this->state->epilogue.size() - 1; ++i) {
    ret.push_back(this->state->epilogue[i]);
  }
  return ret;
}

bool CPUTensorizeContextNode::HasInterPath() {
  return (state->inter_path.size() > 0U);
}

te::Operation CPUTensorizeContextNode::InterPathRootOp() {
  CHECK(this->HasInterPath());
  return this->state->inter_path[(int)this->state->inter_path.size() - 1];
}

Array<te::Operation> CPUTensorizeContextNode::InterPathNonRootOps() {
  Array<te::Operation> ret;
  for (int i = 0; i < (int)this->state->inter_path.size() - 1; ++i) {
    ret.push_back(this->state->inter_path[i]);
  }
  return ret;
}

Array<tir::IterVar> CPUTensorizeContextNode::Split(te::Schedule sch,
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

tir::IterVar CPUTensorizeContextNode::FuseAll(te::Schedule sch,
                                               te::Operation op) {
  te::Operation sop = sch[op]->op;
  const te::ComputeOpNode *cop = sop.as<te::ComputeOpNode>();
  Array<tir::IterVar> axis = cop->axis;
  tir::IterVar fused;
  sch[op].fuse(axis, &fused);
  return fused;
}

Array<tir::IterVar>
CPUTensorizeContextNode::FuseAllAndSplit(te::Schedule sch, te::Operation op,
                                          Array<PrimExpr> factors) {
  tir::IterVar fused = this->FuseAll(sch, op);
  Array<tir::IterVar> tiled = this->Split(sch, op, fused, factors);
  return tiled;
}

void CPUTensorizeContextNode::Inline(te::Schedule sch, te::Operation op) {
  sch[op].compute_inline();
}

bool CPUTensorizeContextNode::CanInline(te::Operation op) {
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
CPUTensorizeContextNode::SecondOpOuterInnerSpatialAxis() {
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
CPUTensorizeContextNode::SecondOpOuterInnerReduceAxis() {
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

std::vector<int>
CPUTensorizeContextNode::TensorizeSpatialAxis(const te::Operation &op) {
  CHECK(this->state->tensorize_iters.count(op));
  Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
  std::unordered_map<tir::IterVar, int> spatial_axis2index;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  int num_spatial_axis = (int)cop->axis.size();
  for (int i = 0; i < num_spatial_axis; ++i) {
    spatial_axis2index[cop->axis[i]] = i;
  }
  std::vector<int> ret;
  for (auto iv : iters) {
    if (spatial_axis2index.count(iv)) {
      ret.push_back(spatial_axis2index.at(iv));
    }
  }
  return ret;
}

std::vector<int>
CPUTensorizeContextNode::TensorizeReduceAxis(const te::Operation &op) {
  CHECK(this->state->tensorize_iters.count(op));
  Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
  std::unordered_map<tir::IterVar, int> reduce_axis2index;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  int num_reduce_axis = (int)cop->reduce_axis.size();
  for (int i = 0; i < num_reduce_axis; ++i) {
    reduce_axis2index[cop->reduce_axis[i]] = i;
  }
  std::vector<int> ret;
  for (auto iv : iters) {
    if (reduce_axis2index.count(iv)) {
      ret.push_back(reduce_axis2index.at(iv));
    }
  }
  return ret;
}

bool CPUTensorizeContextNode::ValidTensorizeFusion(
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

std::vector<int> CPUTensorizeContextNode::GetSpatialExtentsByIndex(
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

std::vector<int> CPUTensorizeContextNode::GetReduceExtentsByIndex(
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

bool CPUTensorizeContextNode::IsInInterPath(const te::Operation &op) {
  for (auto x : this->state->inter_path) {
    if (op == x) {
      return true;
    }
  }
  return false;
}

std::vector<int> CPUTensorizeContextNode::GetSpatialExtentsByInferBound(
    te::Schedule sch, const te::Operation &op) {
  te::Schedule norm_sch = sch.normalize();
  Map<tir::IterVar, Range> bound = te::InferBound(norm_sch);
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  std::vector<int> ret;
  for (auto iv : cop->axis) {
    CHECK(bound.count(iv));
    PrimExpr extent = bound.at(iv)->extent;
    const IntImmNode *as_int = extent.as<IntImmNode>();
    CHECK(as_int) << "Can't infer constant range during scheduling.\n";
    ret.push_back(as_int->value);
  }
  return ret;
}

std::vector<int> CPUTensorizeContextNode::GetReduceExtentsByInferBound(
    te::Schedule sch, const te::Operation &op) {
  te::Schedule norm_sch = sch.normalize();
  Map<tir::IterVar, Range> bound = te::InferBound(norm_sch);
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  std::vector<int> ret;
  for (auto iv : cop->reduce_axis) {
    CHECK(bound.count(iv));
    PrimExpr extent = bound.at(iv)->extent;
    const IntImmNode *as_int = extent.as<IntImmNode>();
    CHECK(as_int) << "Can't infer constant range during scheduling.\n";
    ret.push_back(as_int->value);
  }
  return ret;
}

std::vector<int>
CPUTensorizeContextNode::GetBatchLikeDim(const te::Operation &op) {
  int count_axis = 0;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  CHECK(cop->body.size() == 1) << "Only expect one body.\n";
  IsBatchLikeDim checker;
  std::vector<int> ret;
  for (auto iv : cop->axis) {
    if (checker.is_batch(cop->body[0], iv->var)) {
      ret.push_back(count_axis);
    }
    count_axis += 1;
  }
  return ret;
}

CPUTensorizeContext::CPUTensorizeContext(Layer layer,
                                           TensorizeHyperFusionState state,
                                           hardware::HardwareParam cpu_param) {
  auto node = make_object<CPUTensorizeContextNode>();
  node->layer = layer;
  node->state = state;
  node->cpu_param = cpu_param;
  data_ = node;
}

CPUTensorizeParam::CPUTensorizeParam(int warp_size, int ty_size, int tz_size,
                                       int input_vector_len, int serial_y,
                                       int serial_z, int block_rx, int block_ry,
                                       int block_rz, int warp_rx, int warp_ry,
                                       int warp_rz, int unroll_steps) {
  auto node = make_object<CPUTensorizeParamNode>();
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
  node->unroll_steps = unroll_steps;
  data_ = node;
}

void ScheduleEpilogue(te::Schedule sch, CPUTensorizeContext ctx,
                      CPUTensorizeParam tensorize_param) {
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

void ScheduleSecondOpParallelism(te::Schedule sch, CPUTensorizeContext ctx,
                                 CPUTensorizeParam tensorize_param) {
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
  tensorize_index = ctx->TensorizeSpatialAxis(ctx->state->second_op);
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
  std::vector<int> batch_index = ctx->GetBatchLikeDim(ctx->state->second_op);
  std::unordered_set<int> batch_index_set;
  for (auto ind : batch_index) {
    batch_index_set.insert(ind);
  }
  int z_block_index{-1}, y_block_index{-1};
  for (int i = 0; i < (int)outer_extents.size(); ++i) {
    if (batch_index_set.count(i)) {
      // leave batch-like dims to blockIdx.z
      continue;
    }
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
    ctx->ty_used = true;
    ever_bind = true; // form a valid CPU kernel
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
      ctx->ty_used = true;
      ever_bind = true; // form a valid CPU kernel
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
    ctx->tz_used = true;
    ever_bind = true; // form a valid CPU kernel
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
      ctx->tz_used = true;
      ever_bind = true; // form a valid CPU kernel
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
    tir::IterVar kernel_scope, org;
    sch[second_tensor].split_by_nparts(bind_block_z[0], 1, &kernel_scope, &org);
    sch[second_tensor].pragma(kernel_scope, "auto_unroll_max_step",
                              tensorize_param->unroll_steps);
    sch[second_tensor].pragma(kernel_scope, "unroll_explicit", 1);
    bind_block_z[0] = org;
    sch[second_tensor].fuse(bind_block_z, &fused_block_z);
    sch[second_tensor].bind(fused_block_z,
                            te::thread_axis(Range(), "blockIdx.z"));
    ever_bind = true; // form a valid CPU kernel
  }
  CHECK(ever_bind) << "The scheduler can't bind any axis for CPU.\n";
  // tensorize
  sch[second_tensor].tensorize(tensor_iters[0], pintrin->store_intrinsic);
  /*
   * Find postion to compute at
   */
  tir::IterVar frag_attach_axis;
  for (auto list : {inner_outer, outer_inner, outer_outer}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        frag_attach_axis = list[i];
        break;
      }
    }
    if (frag_attach_axis.defined()) {
      break;
    }
  }
  if (!frag_attach_axis.defined()) {
    frag_attach_axis = fused_block_z;
  }
  CHECK(frag_attach_axis.defined())
      << "Can't find second_frag compute_at position during scheduling.\n";
  tir::IterVar path_attach_axis;
  for (auto list : {outer_outer}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        path_attach_axis = list[i];
        break;
      }
    }
    if (path_attach_axis.defined()) {
      break;
    }
  }
  if (!path_attach_axis.defined()) {
    path_attach_axis = fused_block_z;
  }
  CHECK(path_attach_axis.defined())
      << "Can't find inter path compute_at position during scheduling.\n";
  /*
   * Store the context for following schedules
   */
  ctx->second_op_compute_axis = frag_attach_axis;
  ctx->second_frag = second_frag;
  ctx->path_attach_tensor = second_tensor;
  ctx->path_attach_axis = path_attach_axis;
}

void ScheduleSecondOpLocality(te::Schedule sch, CPUTensorizeContext ctx,
                              CPUTensorizeParam tensorize_param) {
  te::Tensor second_frag = ctx->second_frag;
  CHECK(second_frag.defined() && sch[second_frag]->op.defined());
  const te::ComputeOpNode *frag_cop =
      sch[second_frag]->op.as<te::ComputeOpNode>();
  CHECK(frag_cop != nullptr)
      << second_frag << " is from " << sch[second_frag]->op << "\n";
  // compute_at
  sch[second_frag].compute_at(sch[ctx->state->second_op],
                              ctx->second_op_compute_axis);
  /*
   * Tile the outer reduce axis with largest extent
   */
  // find the reduce outer and inner axis
  std::vector<int> outer_index, inner_index, tensorize_index;
  std::tie(outer_index, inner_index) = ctx->SecondOpOuterInnerReduceAxis();
  tensorize_index = ctx->TensorizeReduceAxis(ctx->state->second_op);
  CHECK(ctx->ValidTensorizeFusion(inner_index, tensorize_index))
      << "The fusion and tensorize decisions are not valid.\n";
  inner_index.erase(inner_index.begin() +
                        (inner_index.size() - tensorize_index.size()),
                    inner_index.end());
  // outer_index should be split into 3 parts
  // inner_index should be placed at itermediate position
  // tensorize_index should be placed at innermost position
  std::vector<int> outer_extents, inner_extents;
  outer_extents =
      ctx->GetReduceExtentsByIndex(ctx->state->second_op, outer_index);
  inner_extents =
      ctx->GetReduceExtentsByIndex(ctx->state->second_op, inner_index);
  // The largest outer axis is bind to a virtual reduce dim y
  int split_id{-1}, largest_dim{-1};
  for (int i = 0; i < (int)outer_extents.size(); ++i) {
    if (outer_extents[i] > largest_dim) {
      largest_dim = outer_extents[i];
      split_id = i;
    }
  }
  // use these vectors to store intermediate axis
  std::vector<tir::IterVar> outer_outer(outer_index.size(), tir::IterVar()),
      outer_inner(outer_index.size(), tir::IterVar()),
      outer_inner_inner(outer_index.size(), tir::IterVar()), inner,
      tensor_iters;
  // fill the inner and tensorize iters
  for (auto ind : inner_index) {
    CHECK(ind < (int)frag_cop->reduce_axis.size());
    inner.push_back(frag_cop->reduce_axis[ind]);
  }
  for (auto ind : tensorize_index) {
    CHECK(ind < (int)frag_cop->reduce_axis.size());
    tensor_iters.push_back(frag_cop->reduce_axis[ind]);
  }
  // split the largets outer reduce axis
  if (split_id >= 0) {
    CHECK(outer_index[split_id] < (int)frag_cop->reduce_axis.size());
    tir::IterVar axis = frag_cop->reduce_axis[outer_index[split_id]];
    Array<tir::IterVar> tiled =
        ctx->Split(sch, second_frag->op, axis,
                   {-1, tensorize_param->block_ry, tensorize_param->warp_ry});
    outer_outer[split_id] = tiled[0];
    outer_inner[split_id] = tiled[1];
    outer_inner_inner[split_id] = tiled[2];
  }
  for (int i = 0; i < (int)outer_extents.size(); ++i) {
    if (i != split_id) {
      outer_outer[i] = frag_cop->reduce_axis[outer_index[i]];
    }
  }
  // reorder
  Array<tir::IterVar> order;
  for (auto list : {outer_outer, outer_inner, inner, outer_inner_inner}) {
    for (auto iv : list) {
      if (iv.defined()) {
        order.push_back(iv);
      }
    }
  }
  // add the remaining spatial axis
  for (auto iv : frag_cop->axis) {
    order.push_back(iv);
  }
  // add the tensorize iters
  for (auto iv : tensor_iters) {
    order.push_back(iv);
  }
  sch[second_frag].reorder(order);
  // tensorize
  CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->second_op));
  PackedIntrinsic pintrin =
      ctx->state->tensorize_intrinsics.at(ctx->state->second_op);
  std::vector<int> tensorize_spatial_index =
      ctx->TensorizeSpatialAxis(ctx->state->second_op);
  CHECK((int)frag_cop->axis.size() > tensorize_spatial_index[0]);
  sch[second_frag].tensorize(frag_cop->axis[tensorize_spatial_index[0]],
                             pintrin->compute_intrinsic);
  /*
   * Store the context for following schedule
   */
  // inter path compute_at position
  tir::IterVar path_attach_axis;
  for (auto list : {outer_outer}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        path_attach_axis = list[i];
        break;
      }
    }
    if (path_attach_axis.defined()) {
      break;
    }
  }
  if (path_attach_axis.defined()) {
    // update the inter path attach info
    ctx->path_attach_tensor = second_frag;
    ctx->path_attach_axis = path_attach_axis;
  }
  // input shared memory compute_at position
  tir::IterVar prologue_attach_axis;
  for (auto list : {outer_outer, outer_inner, inner, outer_inner_inner}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        prologue_attach_axis = list[i];
        break;
      }
    }
    if (prologue_attach_axis.defined()) {
      break;
    }
  }
  CHECK(prologue_attach_axis.defined())
      << "Can't find second prologue shared compute_at position during "
         "scheduling.\n";
  ctx->second_prologue_shared_attach_tensor = second_frag;
  ctx->second_prologue_shared_attach_axis = prologue_attach_axis;
  // input fragment compute_at position
  tir::IterVar frag_attach_axis;
  for (auto list : {inner, outer_inner}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        frag_attach_axis = list[i];
        break;
      }
    }
    if (frag_attach_axis.defined()) {
      break;
    }
  }
  CHECK(frag_attach_axis.defined())
      << "Can't find second prologue fragment compute_at position during "
         "scheduling.\n";
  ctx->second_frag_attach_tensor = second_frag;
  ctx->second_frag_attach_axis = frag_attach_axis;
}

void ScheduleInputFrag(te::Schedule sch, CPUTensorizeContext ctx,
                       CPUTensorizeParam tensorize_param,
                       te::Operation consumer, te::Tensor inp,
                       te::Tensor attach_tensor, tir::IterVar attach_axis,
                       String scope, te::TensorIntrin intrinsic) {
  CHECK(sch[consumer]->op.defined());
  const te::ComputeOpNode *frag_cop = sch[consumer]->op.as<te::ComputeOpNode>();
  CHECK(frag_cop != nullptr);
  te::Tensor inp_frag = sch.cache_read(inp, scope, {consumer});
  CHECK(attach_tensor.defined() && attach_axis.defined());
  sch[inp_frag].compute_at(sch[attach_tensor], attach_axis);
  const te::ComputeOpNode *inp_cop = inp_frag->op.as<te::ComputeOpNode>();
  // tensorize
  const te::ComputeOpNode *cop = intrinsic->op.as<te::ComputeOpNode>();
  int num_axis = (int)cop->axis.size();
  sch[inp_frag].tensorize(inp_cop->axis[(int)inp_cop->axis.size() - num_axis],
                          intrinsic);
}

void ScheduleInputSharedFrag(te::Schedule sch, CPUTensorizeContext ctx,
                             CPUTensorizeParam tensorize_param,
                             te::Operation consumer, te::Tensor inp,
                             te::Tensor shared_attach_tensor,
                             tir::IterVar shared_attach_axis,
                             te::Tensor frag_attach_tensor,
                             tir::IterVar frag_attach_axis, String scope,
                             te::TensorIntrin intrinsic) {
  CHECK(sch[consumer]->op.defined());
  const te::ComputeOpNode *frag_cop = sch[consumer]->op.as<te::ComputeOpNode>();
  CHECK(frag_cop != nullptr);
  te::Tensor inp_shared = sch.cache_read(inp, "shared", {consumer});
  te::Tensor inp_frag = sch.cache_read(inp_shared, scope, {consumer});
  // compute_at shared
  CHECK(shared_attach_tensor.defined() && shared_attach_axis.defined());
  sch[inp_shared].compute_at(sch[shared_attach_tensor], shared_attach_axis);
  // cooperative fetching
  Array<PrimExpr> factors;
  int cur_id{0}, tz_id{-1}, ty_id{-1}, tx_id{-1}, vec_id{-1};
  factors.push_back(-1);
  cur_id += 1;
  if (ctx->tz_used) {
    factors.push_back(tensorize_param->tz_size);
    tz_id = cur_id;
    cur_id += 1;
  }
  if (ctx->ty_used) {
    factors.push_back(tensorize_param->ty_size);
    ty_id = cur_id;
    cur_id += 1;
  }
  tx_id = cur_id;
  factors.push_back(tensorize_param->warp_size);
  cur_id += 1;
  factors.push_back(tensorize_param->input_vector_len);
  vec_id = cur_id;
  Array<tir::IterVar> tiled =
      ctx->FuseAllAndSplit(sch, inp_shared->op, factors);
  if (tz_id >= 0) {
    sch[inp_shared].bind(tiled[tz_id], te::thread_axis(Range(), "threadIdx.z"));
  }
  if (ty_id >= 0) {
    sch[inp_shared].bind(tiled[ty_id], te::thread_axis(Range(), "threadIdx.y"));
  }
  if (tx_id >= 0) {
    sch[inp_shared].bind(tiled[tx_id], te::thread_axis(Range(), "threadIdx.x"));
  }
  if (vec_id >= 0) {
    sch[inp_shared].vectorize(tiled[vec_id]);
  }

  // compute_at frag
  CHECK(frag_attach_tensor.defined() && frag_attach_axis.defined());
  sch[inp_frag].compute_at(sch[frag_attach_tensor], frag_attach_axis);
  const te::ComputeOpNode *inp_cop = inp_frag->op.as<te::ComputeOpNode>();
  // tensorize
  const te::ComputeOpNode *cop = intrinsic->op.as<te::ComputeOpNode>();
  int num_axis = (int)cop->axis.size();
  sch[inp_frag].tensorize(inp_cop->axis[(int)inp_cop->axis.size() - num_axis],
                          intrinsic);
}

void ScheduleInterPath(te::Schedule sch, CPUTensorizeContext ctx,
                       CPUTensorizeParam tensorize_param) {
  te::Operation root_op = ctx->InterPathRootOp();
  sch[root_op].set_scope("shared");
  CHECK(ctx->path_attach_tensor.defined() && ctx->path_attach_axis.defined());
  sch[root_op].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
  for (auto op : ctx->InterPathNonRootOps()) {
    CHECK(ctx->CanInline(op));
    sch[op].compute_inline();
  }
  // cooperative fetching
  Array<PrimExpr> factors;
  int cur_id{0}, tz_id{-1}, ty_id{-1}, tx_id{-1}, vec_id{-1};
  factors.push_back(-1);
  cur_id += 1;
  if (ctx->tz_used) {
    factors.push_back(tensorize_param->tz_size);
    tz_id = cur_id;
    cur_id += 1;
  }
  if (ctx->ty_used) {
    factors.push_back(tensorize_param->ty_size);
    ty_id = cur_id;
    cur_id += 1;
  }
  tx_id = cur_id;
  factors.push_back(tensorize_param->warp_size);
  cur_id += 1;
  factors.push_back(tensorize_param->input_vector_len);
  vec_id = cur_id;
  Array<tir::IterVar> tiled = ctx->FuseAllAndSplit(sch, root_op, factors);
  if (tz_id >= 0) {
    sch[root_op].bind(tiled[tz_id], te::thread_axis(Range(), "threadIdx.z"));
  }
  if (ty_id >= 0) {
    sch[root_op].bind(tiled[ty_id], te::thread_axis(Range(), "threadIdx.y"));
  }
  if (tx_id >= 0) {
    sch[root_op].bind(tiled[tx_id], te::thread_axis(Range(), "threadIdx.x"));
  }
  if (vec_id >= 0) {
    sch[root_op].vectorize(tiled[vec_id]);
  }
}

void ScheduleFirstOpLocality(te::Schedule sch, CPUTensorizeContext ctx,
                             CPUTensorizeParam tensorize_param) {
  te::Operation first_op = ctx->state->first_op;
  CHECK(first_op->num_outputs() == 1)
      << "Only expect one output from first op.\n";
  te::Tensor first_out = first_op.output(0);
  te::Operation consumer;
  if (ctx->HasEpilogue()) {
    Array<te::Operation> path_ops = ctx->InterPathNonRootOps();
    if (path_ops.size() > 0U) {
      consumer = path_ops[0];
    } else {
      consumer = ctx->InterPathRootOp();
    }
  } else {
    CHECK(ctx->second_frag.defined())
        << "The second fragment is not defined.\n";
    consumer = ctx->second_frag->op;
  }
  te::Tensor shared = sch.cache_read(first_out, "shared", {consumer});
  const te::ComputeOpNode *shared_cop = shared->op.as<te::ComputeOpNode>();
  CHECK(shared_cop);
  // compute_at shared
  CHECK(ctx->path_attach_tensor.defined() && ctx->path_attach_axis.defined());
  sch[shared].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
  // get intrinsic
  CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->first_op));
  PackedIntrinsic pintrin =
      ctx->state->tensorize_intrinsics.at(ctx->state->first_op);
  const te::ComputeOpNode *intrin_cop =
      pintrin->store_intrinsic->op.as<te::ComputeOpNode>();
  int num_tensorize_iters = (int)intrin_cop->axis.size();
  // split the remaining spatial axis
  int tz_id{-1}, ty_id{-1};
  std::vector<int> spatial_extents =
      ctx->GetSpatialExtentsByInferBound(sch, shared->op);
  for (int i = 0; i < (int)spatial_extents.size() - num_tensorize_iters; ++i) {
    if ((spatial_extents[i] >= tensorize_param->tz_size) && (ctx->tz_used) &&
        (tz_id < 0)) {
      tz_id = i;
    } else if ((spatial_extents[i] >= tensorize_param->ty_size) &&
               (ctx->ty_used) && (ty_id < 0)) {
      ty_id = i;
    }
  }
  std::vector<tir::IterVar> outers(shared_cop->axis.size(), tir::IterVar()),
      inners(shared_cop->axis.size(), tir::IterVar());
  if (tz_id >= 0) {
    CHECK(tz_id < (int)shared_cop->axis.size());
    tir::IterVar axis = shared_cop->axis[tz_id];
    tir::IterVar outer, inner;
    sch[shared].split_by_nparts(axis, tensorize_param->tz_size, &outer, &inner);
    sch[shared].bind(outer, te::thread_axis(Range(), "threadIdx.z"));
    outers[tz_id] = outer;
    inners[tz_id] = inner;
  }
  if (ty_id >= 0) {
    CHECK(ty_id < (int)shared_cop->axis.size());
    tir::IterVar axis = shared_cop->axis[ty_id];
    tir::IterVar outer, inner;
    sch[shared].split_by_nparts(axis, tensorize_param->ty_size, &outer, &inner);
    sch[shared].bind(outer, te::thread_axis(Range(), "threadIdx.y"));
    outers[ty_id] = outer;
    inners[ty_id] = inner;
  }
  for (int i = 0; i < (int)shared_cop->axis.size() - num_tensorize_iters; ++i) {
    if ((i != tz_id) && (i != ty_id)) {
      outers[i] = shared_cop->axis[i];
    }
  }
  // reorder
  Array<tir::IterVar> order;
  for (auto list : {outers, inners}) {
    for (auto iv : list) {
      if (iv.defined()) {
        order.push_back(iv);
      }
    }
  }
  sch[shared].reorder(order);
  // tensorize
  sch[shared].tensorize(
      shared_cop->axis[(int)shared_cop->axis.size() - num_tensorize_iters],
      pintrin->store_intrinsic);

  // handle fragment
  const te::ComputeOpNode *first_cop = first_op.as<te::ComputeOpNode>();
  // find attach axis
  tir::IterVar frag_attach_axis;
  for (auto list : {outers, inners}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        frag_attach_axis = list[i];
        break;
      }
    }
    if (frag_attach_axis.defined()) {
      break;
    }
  }
  CHECK(frag_attach_axis.defined())
      << "Can't find first_frag compute_at position during scheduling.\n";
  CHECK(ctx->state->tensorize_intrinsics.count(first_op));
  sch[first_op].set_scope(pintrin->compute_scope);
  sch[first_op].compute_at(sch[shared], frag_attach_axis);
  std::vector<int> spatial_index, tensorize_spatial_index, reduce_index,
      tensorize_reduce_index;
  // spatial
  tensorize_spatial_index = ctx->TensorizeSpatialAxis(ctx->state->first_op);
  for (int i = 0; i < (int)first_cop->axis.size(); ++i) {
    spatial_index.push_back(i);
  }
  CHECK(ctx->ValidTensorizeFusion(spatial_index, tensorize_spatial_index))
      << "The fusion and tensorize decisions are not valid.\n";
  spatial_index.erase(spatial_index.begin() + (spatial_index.size() -
                                               tensorize_spatial_index.size()),
                      spatial_index.end());
  // reduce
  tensorize_reduce_index = ctx->TensorizeReduceAxis(ctx->state->first_op);
  for (int i = 0; i < (int)first_cop->reduce_axis.size(); ++i) {
    reduce_index.push_back(i);
  }
  CHECK(ctx->ValidTensorizeFusion(reduce_index, tensorize_reduce_index))
      << "The fusion and tensorize decisions are not valid.\n";
  reduce_index.erase(reduce_index.begin() +
                         (reduce_index.size() - tensorize_reduce_index.size()),
                     reduce_index.end());
  // find the largest one
  std::vector<int> reduce_extents =
      ctx->GetReduceExtentsByIndex(ctx->state->first_op, reduce_index);
  int split_id{-1}, split_extent{-1};
  for (int i = 0; i < (int)reduce_extents.size(); ++i) {
    if (reduce_extents[i] > split_extent) {
      split_extent = reduce_extents[i];
      split_id = i;
    }
  }
  std::vector<tir::IterVar> outs(reduce_extents.size(), tir::IterVar()),
      medians(reduce_extents.size(), tir::IterVar()),
      ins(reduce_extents.size(), tir::IterVar());
  if (split_id >= 0) {
    tir::IterVar axis = first_cop->reduce_axis[split_id];
    Array<tir::IterVar> tiled =
        ctx->Split(sch, first_op, axis,
                   {-1, tensorize_param->block_rx, tensorize_param->warp_rx});
    outs[split_id] = tiled[0];
    medians[split_id] = tiled[1];
    ins[split_id] = tiled[2];
  }
  for (int i = 0; i < (int)reduce_extents.size(); ++i) {
    if (i != split_id) {
      outs[i] = first_cop->reduce_axis[i];
    }
  }
  Array<tir::IterVar> new_order;
  for (auto list : {outs, medians, ins}) {
    for (auto iv : list) {
      if (iv.defined()) {
        new_order.push_back(iv);
      }
    }
  }
  for (auto ind : spatial_index) {
    new_order.push_back(first_cop->axis[ind]);
  }
  for (auto ind : tensorize_spatial_index) {
    new_order.push_back(first_cop->axis[ind]);
  }
  for (auto ind : tensorize_reduce_index) {
    new_order.push_back(first_cop->reduce_axis[ind]);
  }
  sch[first_op].reorder(new_order);
  // tensorize
  CHECK(tensorize_spatial_index[0] < (int)first_cop->axis.size());
  sch[first_op].tensorize(first_cop->axis[tensorize_spatial_index[0]],
                          pintrin->compute_intrinsic);
  /*
   * Store the context for following schedules
   */
  tir::IterVar prologue_frag_attach_axis;
  for (auto list : {medians}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        prologue_frag_attach_axis = list[i];
        break;
      }
    }
    if (prologue_frag_attach_axis.defined()) {
      break;
    }
  }
  CHECK(prologue_frag_attach_axis.defined())
      << "Can't find first_op's prologue fragment compute_at position during "
         "scheduling.\n";
  ctx->first_frag_attach_tensor = first_op.output(0);
  ctx->first_frag_attach_axis = prologue_frag_attach_axis;
  tir::IterVar prologue_shared_attach_axis;
  for (auto list : {outs}) {
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      if (list[i].defined()) {
        prologue_shared_attach_axis = list[i];
        break;
      }
    }
    if (prologue_shared_attach_axis.defined()) {
      break;
    }
  }
  CHECK(prologue_shared_attach_axis.defined())
      << "Can't find first_op's prologue shared compute_at position during "
         "scheduling.\n";
  ctx->first_prologue_shared_attach_tensor = first_op.output(0);
  ctx->first_prologue_shared_attach_axis = prologue_shared_attach_axis;
}

te::Schedule TensorizeCPU(Layer layer, TensorizeHyperFusionState state,
                           hardware::HardwareParam cpu_param,
                           CPUTensorizeParam tensorize_param) {
  te::Schedule sch = te::create_schedule(layer->ops);
  CPUTensorizeContext ctx = CPUTensorizeContext(layer, state, cpu_param);
  /*
   * Schedule epilogue
   * Currently, we treat epilogue as a separate kernel
   */
  ScheduleEpilogue(sch, ctx, tensorize_param);
  /*
   * Schedule second op tensor
   * The second tensor determines the overall parallelism
   */
  ScheduleSecondOpParallelism(sch, ctx, tensorize_param);
  /*
   * Schedule second op fragment
   * The second frag determines part of the locality
   */
  ScheduleSecondOpLocality(sch, ctx, tensorize_param);
  /*
   * Schedule second op's inputs, including inter path
   * The inter path's attach position can be different from
   * other inputs'.
   */
  CHECK(ctx->second_frag.defined() && ctx->second_frag->op.defined());
  int count_num_input = 0;
  for (auto inp : ctx->second_frag->op->InputTensors()) {
    CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->second_op));
    PackedIntrinsic pintrin =
        ctx->state->tensorize_intrinsics.at(ctx->state->second_op);
    CHECK(count_num_input < (int)pintrin->load_scopes.size());
    if ((inp->op == ctx->state->first_op) || ctx->IsInInterPath(inp->op)) {
      // input from first op or inter path
      ScheduleInputFrag(sch, ctx, tensorize_param, ctx->second_frag->op, inp,
                        ctx->second_frag_attach_tensor,
                        ctx->second_frag_attach_axis,
                        pintrin->load_scopes[count_num_input],
                        pintrin->load_intrinsics[count_num_input]);
    } else {
      // input from prologue
      ScheduleInputSharedFrag(sch, ctx, tensorize_param, ctx->second_frag->op,
                              inp, ctx->second_prologue_shared_attach_tensor,
                              ctx->second_prologue_shared_attach_axis,
                              ctx->second_frag_attach_tensor,
                              ctx->second_frag_attach_axis,
                              pintrin->load_scopes[count_num_input],
                              pintrin->load_intrinsics[count_num_input]);
    }
    count_num_input += 1;
  }
  /*
   * Schedule second op's prologue
   */
  for (auto op_list : ctx->state->second_op_prologue) {
    for (auto op : op_list) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
    }
  }
  /*
   * Schedule inter path
   */
  ScheduleInterPath(sch, ctx, tensorize_param);
  /*
   * Schedule first op
   */
  ScheduleFirstOpLocality(sch, ctx, tensorize_param);
  /*
   * Schedule first op's inputs
   */
  count_num_input = 0;
  for (auto inp : ctx->state->first_op->InputTensors()) {
    CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->first_op));
    PackedIntrinsic pintrin =
        ctx->state->tensorize_intrinsics.at(ctx->state->first_op);
    CHECK(count_num_input < (int)pintrin->load_scopes.size());
    // input from prologue
    ScheduleInputSharedFrag(
        sch, ctx, tensorize_param, ctx->state->first_op, inp,
        ctx->first_prologue_shared_attach_tensor,
        ctx->first_prologue_shared_attach_axis, ctx->first_frag_attach_tensor,
        ctx->first_frag_attach_axis, pintrin->load_scopes[count_num_input],
        pintrin->load_intrinsics[count_num_input]);
    count_num_input += 1;
  }
  /*
   * Schedule first op's prologue
   */
  for (auto op_list : ctx->state->first_op_prologue) {
    for (auto op : op_list) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
    }
  }
  return sch;
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CPUTensorizeContext")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                       hardware::HardwareParam cpu_param) {
      return CPUTensorizeContext(layer, state, cpu_param);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CPUTensorizeParam")
    .set_body_typed([](int warp_size, int ty_size, int tz_size,
                       int input_vector_len, int serial_y, int serial_z,
                       int block_rx, int block_ry, int block_rz, int warp_rx,
                       int warp_ry, int warp_rz, int unroll_steps) {
      return CPUTensorizeParam(warp_size, ty_size, tz_size, input_vector_len,
                                serial_y, serial_z, block_rx, block_ry,
                                block_rz, warp_rx, warp_ry, warp_rz,
                                unroll_steps);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeCPU")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                       hardware::HardwareParam cpu_param,
                       CPUTensorizeParam tensorize_param) {
      return TensorizeCPU(layer, state, cpu_param, tensorize_param);
    });

} // namespace auto_tensorize

} // namespace ditto