#include <auto_compute/graph.h>
#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/searchDriver.h>
#include <auto_tensorize/hyper_fusion.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/state.h>
#include <tvm/te/schedule_pass.h>
#include <globals.h>

#define MAX_CANDIDATES_NUMBER 10000
#define SURVEY_CANDIDATES_NUMBER 100
#define SELECT_TOP 10

namespace ditto
{

  namespace auto_tensorize
  {

    TVM_REGISTER_NODE_TYPE(FusionChoiceNode);
    TVM_REGISTER_NODE_TYPE(MatchInfoNode);
    TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);
    TVM_REGISTER_NODE_TYPE(CUDATensorizeContextNode);
    TVM_REGISTER_NODE_TYPE(CUDATensorizeParamNode);

    FusionChoice::FusionChoice(te::Operation first_op, te::Operation second_op,
                               Array<tir::IterVar> ordered_iters, int attach_pos,
                               Array<IntImm> secondOpOuterTilingFactors,
                               Array<IntImm> secondOpOuterIndices,
                               FusionItem fusionItem, FusionResult fusionResult)
    {
      auto node = make_object<FusionChoiceNode>();
      node->first_op = first_op;
      node->second_op = second_op;
      node->ordered_iters = ordered_iters;
      node->attach_pos = attach_pos;
      node->secondOpOuterIndices = secondOpOuterIndices;
      node->secondOpOuterTilingFactors = secondOpOuterTilingFactors;
      node->fusionItem = fusionItem;
      node->fusionResult = fusionResult;
      data_ = node;
    }

    MatchInfo::MatchInfo(Array<tir::IterVar> axis, PackedIntrinsic intrin, const tir::StringImm impl)
    {
      auto node = make_object<MatchInfoNode>();
      node->axis = axis;
      node->intrin = intrin;
      node->impl = impl;
      data_ = node;
    }

    TensorizeHyperFusionState::TensorizeHyperFusionState(
        Layer layer, FusionChoice fuse_choice,
        Map<te::Operation, MatchInfo> match_info)
    {
      auto node = make_object<TensorizeHyperFusionStateNode>();
      //   Array<Array<te::Operation>> first_op_prologue;
      //   Array<Array<te::Operation>> second_op_prologue;
      //   Array<te::Operation> inter_path;
      //   Array<te::Operation> epilogue;
      node->layer = layer;
      node->first_op = fuse_choice->first_op;
      node->second_op = fuse_choice->second_op;

      CHECK(fuse_choice->attach_pos < (int)fuse_choice->ordered_iters.size());
      for (int i = 0; i <= fuse_choice->attach_pos; ++i)
      {
        tir::IterVar iv = fuse_choice->ordered_iters[i];
        if (iv->iter_type == tir::IterVarType::kDataPar)
        {
          node->fused_spatial_outer_iters.push_back(iv);
        }
        else if (iv->iter_type == tir::IterVarType::kCommReduce)
        {
          node->fused_reduce_outer_iters.push_back(iv);
        }
        else
        {
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
        if (!cop)
        {
          return std::make_pair(false, false);
        }
        CHECK(!prefix_array.count(op))
            << "The same op " << op << " is visited multiple times.\n"
            << "The given layer is not a tree structure.\n"
            << "Layer information:\n"
            << layer;
        bool met_first{false}, met_second{false};
        if (op == node->first_op)
        {
          for (auto inp : cop->InputTensors())
          {
            bool inp_met_first{false}, inp_met_second{false};
            std::tie(inp_met_first, inp_met_second) = helper(inp->op);
            CHECK((!first_op_found) && (!second_op_found));
            CHECK((!inp_met_first) && (!inp_met_second));
            if (prefix_array.count(inp->op))
            {
              node->first_op_prologue.push_back(prefix_array.at(inp->op));
            }
          }
          // globally update the first_op_found
          first_op_found = true;
          met_first = true;
        }
        else if (op == node->second_op)
        {
          bool first_op_in_producer{false};
          for (auto inp : cop->InputTensors())
          {
            bool inp_met_first{false}, inp_met_second{false};
            std::tie(inp_met_first, inp_met_second) = helper(inp->op);
            CHECK(!inp_met_second);
            CHECK(first_op_found && (!second_op_found)) << "first op found: " << first_op_found << " second op found: " << second_op_found << std::endl;
            // update to see if first_op is in producers
            first_op_in_producer |= inp_met_first;
            if (prefix_array.count(inp->op))
            {
              if (inp_met_first)
              {
                // belong to path
                CHECK(node->inter_path.size() == 0U)
                    << "Multiple paths from first op to second op.\n";
                node->inter_path = prefix_array.at(inp->op);
              }
              else
              {
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
        }
        else
        {
          for (auto inp : cop->InputTensors())
          {
            bool inp_met_first{false}, inp_met_second{false};
            std::tie(inp_met_first, inp_met_second) = helper(inp->op);
            met_first |= inp_met_first;
            met_second |= inp_met_second;
            if (prefix_array.count(inp->op))
            {
              for (auto v : prefix_array.at(inp->op))
              {
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
      if (prefix_array.count(layer->ops[0]))
      {
        node->epilogue = prefix_array.at(layer->ops[0]);
      }

      // CHECK(match_info.size() == 2U);
      CHECK(match_info.count(node->first_op) && match_info.count(node->second_op));
      node->tensorize_iters.Set(node->first_op,
                                match_info.at(node->first_op)->axis);
      node->tensorize_intrinsics.Set(node->first_op,
                                     match_info.at(node->first_op)->intrin);
      node->tensorize_iters.Set(node->second_op,
                                match_info.at(node->second_op)->axis);
      node->tensorize_intrinsics.Set(node->second_op,
                                     match_info.at(node->second_op)->intrin);
      for (auto a_match_info: match_info){
        CHECK(a_match_info.second->axis.defined());
        node->tensorize_iters.Set(
          a_match_info.first,
          a_match_info.second->axis
        );
        CHECK(a_match_info.second->intrin.defined());
        node->tensorize_intrinsics.Set(
          a_match_info.first,
          a_match_info.second->intrin
        );
        CHECK(a_match_info.second->impl.defined());
        node->tensorize_impl.Set(
          a_match_info.first,
          a_match_info.second->impl
        );
      }

      node->secondOpOuterTileFactors = fuse_choice->secondOpOuterTilingFactors;
      node->secondOpOuterIndices = fuse_choice->secondOpOuterIndices;
      data_ = node;
    }

    bool CUDATensorizeContextNode::HasEpilogue()
    {
      return (state->epilogue.size() > 0U);
    }

    te::Operation CUDATensorizeContextNode::EpilogueRootOp()
    {
      CHECK(this->HasEpilogue());
      return this->state->epilogue[(int)this->state->epilogue.size() - 1];
    }

    Array<te::Operation> CUDATensorizeContextNode::EpilogueNonRootOps()
    {
      Array<te::Operation> ret;
      for (int i = 0; i < (int)this->state->epilogue.size() - 1; ++i)
      {
        ret.push_back(this->state->epilogue[i]);
      }
      return ret;
    }

    bool CUDATensorizeContextNode::HasInterPath()
    {
      return (state->inter_path.size() > 0U);
    }

    te::Operation CUDATensorizeContextNode::InterPathRootOp()
    {
      CHECK(this->HasInterPath());
      return this->state->inter_path[(int)this->state->inter_path.size() - 1];
    }

    Array<te::Operation> CUDATensorizeContextNode::InterPathNonRootOps()
    {
      Array<te::Operation> ret;
      for (int i = 0; i < (int)this->state->inter_path.size() - 1; ++i)
      {
        ret.push_back(this->state->inter_path[i]);
      }
      return ret;
    }

    Array<tir::IterVar> CUDATensorizeContextNode::Split(te::Schedule sch,
                                                        te::Operation op,
                                                        tir::IterVar iv,
                                                        Array<PrimExpr> factors)
    {
      std::vector<tir::IterVar> ret;
      int nparts = (int)factors.size();
      CHECK(nparts > 0);
      for (int i = nparts - 1; i > 0; --i)
      {
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
                                                   te::Operation op)
    {
      te::Operation sop = sch[op]->op;
      const te::ComputeOpNode *cop = sop.as<te::ComputeOpNode>();
      Array<tir::IterVar> axis = cop->axis;
      tir::IterVar fused;
      sch[op].fuse(axis, &fused);
      return fused;
    }

    Array<tir::IterVar>
    CUDATensorizeContextNode::FuseAllAndSplit(te::Schedule sch, te::Operation op,
                                              Array<PrimExpr> factors)
    {
      tir::IterVar fused = this->FuseAll(sch, op);
      Array<tir::IterVar> tiled = this->Split(sch, op, fused, factors);
      return tiled;
    }

    void CUDATensorizeContextNode::Inline(te::Schedule sch, te::Operation op)
    {
      sch[op].compute_inline();
    }

    bool CUDATensorizeContextNode::CanInline(te::Operation op)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      if (!cop)
      {
        return false;
      }
      if (cop->reduce_axis.size() > 0U)
      {
        return false;
      }
      for (auto out : this->layer->ops)
      {
        if (out == op)
        {
          // is output op
          return false;
        }
      }
      return true;
    }

    std::pair<std::vector<int>, std::vector<int>>
    CUDATensorizeContextNode::SecondOpOuterInnerSpatialAxis()
    {
      std::vector<int> outer_index;
      std::vector<int> inner_index;
      std::unordered_map<tir::IterVar, int> spatial_axis2index;
      std::unordered_set<int> outer_set;
      const te::ComputeOpNode *second_cop =
          this->state->second_op.as<te::ComputeOpNode>();
      CHECK(second_cop);
      int num_spatial_axis = (int)second_cop->axis.size();
      for (int i = 0; i < num_spatial_axis; ++i)
      {
        spatial_axis2index[second_cop->axis[i]] = i;
      }
      for (auto iv : this->state->fused_spatial_outer_iters)
      {
        CHECK(spatial_axis2index.count(iv));
        int idx = spatial_axis2index.at(iv);
        outer_index.push_back(idx);
        outer_set.insert(idx);
      }
      for (int i = 0; i < num_spatial_axis; ++i)
      {
        if (!outer_set.count(i))
        {
          inner_index.push_back(i);
        }
      }
      return std::make_pair(outer_index, inner_index);
    }

    std::pair<std::vector<int>, std::vector<int>>
    CUDATensorizeContextNode::SecondOpOuterInnerReduceAxis()
    {
      std::vector<int> outer_index;
      std::vector<int> inner_index;
      std::unordered_map<tir::IterVar, int> reduce_axis2index;
      std::unordered_set<int> outer_set;
      const te::ComputeOpNode *second_cop =
          this->state->second_op.as<te::ComputeOpNode>();
      CHECK(second_cop);
      int num_reduce_axis = (int)second_cop->reduce_axis.size();
      for (int i = 0; i < num_reduce_axis; ++i)
      {
        reduce_axis2index[second_cop->reduce_axis[i]] = i;
      }
      for (auto iv : this->state->fused_reduce_outer_iters)
      {
        CHECK(reduce_axis2index.count(iv));
        int idx = reduce_axis2index.at(iv);
        outer_index.push_back(idx);
        outer_set.insert(idx);
      }
      for (int i = 0; i < num_reduce_axis; ++i)
      {
        if (!outer_set.count(i))
        {
          inner_index.push_back(i);
        }
      }
      return std::make_pair(outer_index, inner_index);
    }

    std::vector<int>
    CUDATensorizeContextNode::TensorizeSpatialAxis(const te::Operation &op)
    {
      CHECK(this->state->tensorize_iters.count(op));
      Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
      std::unordered_map<tir::IterVar, int> spatial_axis2index;
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      int num_spatial_axis = (int)cop->axis.size();
      for (int i = 0; i < num_spatial_axis; ++i)
      {
        spatial_axis2index[cop->axis[i]] = i;
      }
      std::vector<int> ret;
      for (auto iv : iters)
      {
        if (spatial_axis2index.count(iv))
        {
          ret.push_back(spatial_axis2index.at(iv));
        }
      }
      return ret;
    }

    std::vector<int>
    CUDATensorizeContextNode::TensorizeReduceAxis(const te::Operation &op)
    {
      CHECK(this->state->tensorize_iters.count(op));
      Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
      std::unordered_map<tir::IterVar, int> reduce_axis2index;
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      int num_reduce_axis = (int)cop->reduce_axis.size();
      for (int i = 0; i < num_reduce_axis; ++i)
      {
        reduce_axis2index[cop->reduce_axis[i]] = i;
      }
      std::vector<int> ret;
      for (auto iv : iters)
      {
        if (reduce_axis2index.count(iv))
        {
          ret.push_back(reduce_axis2index.at(iv));
        }
      }
      return ret;
    }

    bool CUDATensorizeContextNode::ValidTensorizeFusion(
        const std::vector<int> &inner_index,
        const std::vector<int> &tensorize_index)
    {
      int len1 = (int)inner_index.size();
      int len2 = (int)tensorize_index.size();
      if (len2 > len1)
      {
        return false;
      }
      // the tensorized iters should be innermost loops
      // e.g., inner: [i1, i2, i3, i4], tensorize: [i3, i4]
      for (int i = 0; i < len2; ++i)
      {
        if (inner_index[i + len1 - len2] != tensorize_index[i])
        {
          return false;
        }
      }
      return true;
    }

    std::vector<int> CUDATensorizeContextNode::GetSpatialExtentsByIndex(
        const te::Operation &op, const std::vector<int> &index)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      int num_axis = (int)cop->axis.size();
      std::vector<int> ret;
      for (auto ind : index)
      {
        CHECK(ind < num_axis);
        tir::IterVar iv = cop->axis[ind];
        PrimExpr ext = iv->dom->extent;
        const IntImmNode *as_int = ext.as<IntImmNode>();
        // CHECK(as_int) << "Can't infer constant range during scheduling.\n" << extent;
        if (as_int)
        {
          ret.push_back(as_int->value);
        }
        else
        {
          ret.push_back(-1); // can't infer
        }
      }
      return ret;
    }

    std::vector<int> CUDATensorizeContextNode::GetReduceExtentsByIndex(
        const te::Operation &op, const std::vector<int> &index)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      int num_axis = (int)cop->reduce_axis.size();
      std::vector<int> ret;
      for (auto ind : index)
      {
        CHECK(ind < num_axis);
        tir::IterVar iv = cop->reduce_axis[ind];
        PrimExpr ext = iv->dom->extent;
        const IntImmNode *as_int = ext.as<IntImmNode>();
        CHECK(as_int) << "Currently only static shape is supported.\n";
        ret.push_back(as_int->value);
      }
      return ret;
    }

    bool CUDATensorizeContextNode::IsInInterPath(const te::Operation &op)
    {
      for (auto x : this->state->inter_path)
      {
        if (op == x)
        {
          return true;
        }
      }
      return false;
    }

    std::vector<int> CUDATensorizeContextNode::GetSpatialExtentsByInferBound(
        te::Schedule sch, const te::Operation &op)
    {
      te::Schedule norm_sch = sch.normalize();
      Map<tir::IterVar, Range> bound = te::InferBound(norm_sch);
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      std::vector<int> ret;
      for (auto iv : cop->axis)
      {
        CHECK(bound.count(iv));
        PrimExpr extent = bound.at(iv)->extent;
        const IntImmNode *as_int = extent.as<IntImmNode>();
        CHECK(as_int) << "Can't infer constant range during scheduling.\n"
                      << extent;
        ret.push_back(as_int->value);
      }
      return ret;
    }

    std::vector<int> CUDATensorizeContextNode::GetReduceExtentsByInferBound(
        te::Schedule sch, const te::Operation &op)
    {
      te::Schedule norm_sch = sch.normalize();
      Map<tir::IterVar, Range> bound = te::InferBound(norm_sch);
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      std::vector<int> ret;
      for (auto iv : cop->reduce_axis)
      {
        CHECK(bound.count(iv));
        PrimExpr extent = bound.at(iv)->extent;
        const IntImmNode *as_int = extent.as<IntImmNode>();
        CHECK(as_int) << "Can't infer constant range during scheduling.\n";
        ret.push_back(as_int->value);
      }
      return ret;
    }

    std::vector<int>
    CUDATensorizeContextNode::GetBatchLikeDim(const te::Operation &op)
    {
      int count_axis = 0;
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      CHECK(cop->body.size() == 1) << "Only expect one body.\n";
      IsBatchLikeDim checker;
      std::vector<int> ret;
      for (auto iv : cop->axis)
      {
        if (checker.is_batch(cop->body[0], iv->var))
        {
          ret.push_back(count_axis);
        }
        count_axis += 1;
      }
      return ret;
    }

    CUDATensorizeContext::CUDATensorizeContext(Layer layer,
                                               TensorizeHyperFusionState state,
                                               hardware::HardwareParam cuda_param)
    {
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
                                           int warp_rz, int unroll_steps)
    {
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
      node->unroll_steps = unroll_steps;
      data_ = node;
    }

    void ScheduleEpilogue(te::Schedule sch, CUDATensorizeContext ctx,
                          CUDATensorizeParam tensorize_param)
    {
      te::Operation cur_op;
      if (ctx->HasEpilogue())
      {
        cur_op = ctx->EpilogueRootOp();
        Array<tir::IterVar> tiled =
            ctx->FuseAllAndSplit(sch, cur_op, {-1, tensorize_param->warp_size});
        sch[cur_op].bind(tiled[0], te::thread_axis(Range(), "blockIdx.x"));
        sch[cur_op].bind(tiled[1], te::thread_axis(Range(), "threadIdx.x"));
        Array<te::Operation> remain_ops = ctx->EpilogueNonRootOps();
        // the remaining non-root ops should be inlined
        for (auto op : remain_ops)
        {
          CHECK(ctx->CanInline(op));
          sch[op].compute_inline();
        }
      }
    }

    void ScheduleSecondOpParallelism(te::Schedule sch, CUDATensorizeContext ctx,
                                     CUDATensorizeParam tensorize_param)
    {
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
      for (int i = 0; i < (int)inner_extents.size(); ++i)
      {
        if ((inner_extents[i] >= tensorize_param->tz_size) && z_dim_index < 0)
        {
          z_dim_index = i;
        }
        else if ((inner_extents[i] >= tensorize_param->ty_size) &&
                 y_dim_index < 0)
        {
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
      for (auto ind : batch_index)
      {
        batch_index_set.insert(ind);
      }
      int z_block_index{-1}, y_block_index{-1};
      for (int i = 0; i < (int)outer_extents.size(); ++i)
      {
        if (batch_index_set.count(i))
        {
          // leave batch-like dims to blockIdx.z
          continue;
        }
        if ((outer_extents[i] >= tensorize_param->tz_size) && z_block_index < 0)
        {
          z_block_index = i;
        }
        else if ((outer_extents[i] >= tensorize_param->ty_size) &&
                 y_block_index < 0)
        {
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
      for (auto ind : tensorize_index)
      {
        CHECK((int)cur_cop->axis.size() > ind);
        tensor_iters.push_back(cur_cop->axis[ind]);
      }
      bool ever_bind{false};
      // first, bind thread y
      if (y_dim_index >= 0)
      {
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
        ever_bind = true; // form a valid GPU kernel
        inner_outer[y_dim_index] = outer;
        inner_inner[y_dim_index] = inner;
      }
      if (y_block_index >= 0)
      {
        CHECK((int)outer_index.size() > y_block_index);
        int ind = outer_index[y_block_index];
        CHECK((int)cur_cop->axis.size() > ind);
        tir::IterVar axis = cur_cop->axis[ind];
        if (y_dim_index < 0)
        {
          // use outer axis for block x and thread y
          Array<tir::IterVar> tiled =
              ctx->Split(sch, second_tensor->op, axis,
                         {-1, tensorize_param->ty_size, tensorize_param->serial_y});
          sch[second_tensor].bind(tiled[0], te::thread_axis(Range(), "blockIdx.x"));
          sch[second_tensor].bind(tiled[1],
                                  te::thread_axis(Range(), "threadIdx.y"));
          ctx->ty_used = true;
          ever_bind = true; // form a valid GPU kernel
          outer_outer[y_block_index] = tiled[0];
          outer_inner[y_block_index] = tiled[1];
          outer_inner_inner[y_block_index] = tiled[2];
        }
        else
        {
          sch[second_tensor].bind(axis, te::thread_axis(Range(), "blockIdx.x"));
        }
      }
      // then, bind thread z
      if (z_dim_index >= 0)
      {
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
        ever_bind = true; // form a valid GPU kernel
        inner_outer[z_dim_index] = outer;
        inner_inner[z_dim_index] = inner;
      }
      if (z_block_index >= 0)
      {
        CHECK((int)outer_index.size() > z_block_index);
        int ind = outer_index[z_block_index];
        CHECK((int)cur_cop->axis.size() > ind);
        tir::IterVar axis = cur_cop->axis[ind];
        if (z_dim_index < 0)
        {
          // use outer axis for block y and thread z
          Array<tir::IterVar> tiled =
              ctx->Split(sch, second_tensor->op, axis,
                         {-1, tensorize_param->tz_size, tensorize_param->serial_z});
          sch[second_tensor].bind(tiled[0], te::thread_axis(Range(), "blockIdx.y"));
          sch[second_tensor].bind(tiled[1],
                                  te::thread_axis(Range(), "threadIdx.z"));
          ctx->tz_used = true;
          ever_bind = true; // form a valid GPU kernel
          outer_outer[z_block_index] = tiled[0];
          outer_inner[z_block_index] = tiled[1];
          outer_inner_inner[z_block_index] = tiled[2];
        }
        else
        {
          sch[second_tensor].bind(axis, te::thread_axis(Range(), "blockIdx.y"));
        }
      }
      // finally, bind block z
      std::vector<tir::IterVar> bind_block_z;
      for (int i = 0; i < (int)outer_index.size(); ++i)
      {
        if ((i != y_block_index) && (i != z_block_index))
        {
          // this outer axis is never bound
          bind_block_z.push_back(cur_cop->axis[outer_index[i]]);
        }
      }
      // collect the remaining inner axis
      for (int i = 0; i < (int)inner_index.size(); ++i)
      {
        if ((i != y_dim_index) && (i != z_dim_index))
        {
          // this inner axis is never bound
          inner_outer[i] = cur_cop->axis[inner_index[i]];
        }
      }
      // Reorder all the axis
      Array<tir::IterVar> order;
      for (auto list : {bind_block_z, outer_outer, outer_inner, inner_outer,
                        outer_inner_inner, inner_inner, tensor_iters})
      {
        for (auto iv : list)
        {
          if (iv.defined())
          {
            order.push_back(iv);
          }
        }
      }
      sch[second_tensor].reorder(order);
      tir::IterVar fused_block_z;
      if (bind_block_z.size() > 0U)
      {
        tir::IterVar kernel_scope, org;
        sch[second_tensor].split_by_nparts(bind_block_z[0], 1, &kernel_scope, &org);
        sch[second_tensor].pragma(kernel_scope, "auto_unroll_max_step",
                                  tensorize_param->unroll_steps);
        sch[second_tensor].pragma(kernel_scope, "unroll_explicit", 1);
        bind_block_z[0] = org;
        sch[second_tensor].fuse(bind_block_z, &fused_block_z);
        sch[second_tensor].bind(fused_block_z,
                                te::thread_axis(Range(), "blockIdx.z"));
        ever_bind = true; // form a valid GPU kernel
      }
      CHECK(ever_bind) << "The scheduler can't bind any axis for GPU.\n";
      // tensorize
      sch[second_tensor].tensorize(tensor_iters[0], pintrin->store_intrinsic);
  /*
   * Find postion to compute at
   */
      tir::IterVar frag_attach_axis;
      for (auto list : {inner_outer, outer_inner, outer_outer})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            frag_attach_axis = list[i];
            break;
          }
        }
        if (frag_attach_axis.defined())
        {
          break;
        }
      }
      if (!frag_attach_axis.defined())
      {
        frag_attach_axis = fused_block_z;
      }
      CHECK(frag_attach_axis.defined())
          << "Can't find second_frag compute_at position during scheduling.\n";
      tir::IterVar path_attach_axis;
      for (auto list : {outer_outer})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            path_attach_axis = list[i];
            break;
          }
        }
        if (path_attach_axis.defined())
        {
          break;
        }
      }
      if (!path_attach_axis.defined())
      {
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

    void ScheduleSecondOpLocality(te::Schedule sch, CUDATensorizeContext ctx,
                                  CUDATensorizeParam tensorize_param)
    {
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
      for (int i = 0; i < (int)outer_extents.size(); ++i)
      {
        if (outer_extents[i] > largest_dim)
        {
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
      for (auto ind : inner_index)
      {
        CHECK(ind < (int)frag_cop->reduce_axis.size());
        inner.push_back(frag_cop->reduce_axis[ind]);
      }
      for (auto ind : tensorize_index)
      {
        CHECK(ind < (int)frag_cop->reduce_axis.size());
        tensor_iters.push_back(frag_cop->reduce_axis[ind]);
      }
      // split the largets outer reduce axis
      if (split_id >= 0)
      {
        CHECK(outer_index[split_id] < (int)frag_cop->reduce_axis.size());
        tir::IterVar axis = frag_cop->reduce_axis[outer_index[split_id]];
        Array<tir::IterVar> tiled =
            ctx->Split(sch, second_frag->op, axis,
                       {-1, tensorize_param->block_ry, tensorize_param->warp_ry});
        outer_outer[split_id] = tiled[0];
        outer_inner[split_id] = tiled[1];
        outer_inner_inner[split_id] = tiled[2];
      }
      for (int i = 0; i < (int)outer_extents.size(); ++i)
      {
        if (i != split_id)
        {
          outer_outer[i] = frag_cop->reduce_axis[outer_index[i]];
        }
      }
      // reorder
      Array<tir::IterVar> order;
      for (auto list : {outer_outer, outer_inner, inner, outer_inner_inner})
      {
        for (auto iv : list)
        {
          if (iv.defined())
          {
            order.push_back(iv);
          }
        }
      }
      // add the remaining spatial axis
      for (auto iv : frag_cop->axis)
      {
        order.push_back(iv);
      }
      // add the tensorize iters
      for (auto iv : tensor_iters)
      {
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
      for (auto list : {outer_outer})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            path_attach_axis = list[i];
            break;
          }
        }
        if (path_attach_axis.defined())
        {
          break;
        }
      }
      if (path_attach_axis.defined())
      {
        // update the inter path attach info
        ctx->path_attach_tensor = second_frag;
        ctx->path_attach_axis = path_attach_axis;
      }
      // input shared memory compute_at position
      tir::IterVar prologue_attach_axis;
      for (auto list : {outer_outer, outer_inner, inner, outer_inner_inner})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            prologue_attach_axis = list[i];
            break;
          }
        }
        if (prologue_attach_axis.defined())
        {
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
      for (auto list : {inner, outer_inner})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            frag_attach_axis = list[i];
            break;
          }
        }
        if (frag_attach_axis.defined())
        {
          break;
        }
      }
      CHECK(frag_attach_axis.defined())
          << "Can't find second prologue fragment compute_at position during "
             "scheduling.\n";
      ctx->second_frag_attach_tensor = second_frag;
      ctx->second_frag_attach_axis = frag_attach_axis;
    }

    void ScheduleInputFrag(te::Schedule sch, CUDATensorizeContext ctx,
                           CUDATensorizeParam tensorize_param,
                           te::Operation consumer, te::Tensor inp,
                           te::Tensor attach_tensor, tir::IterVar attach_axis,
                           String scope, te::TensorIntrin intrinsic)
    {
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

    void ScheduleInputSharedFrag(te::Schedule sch, CUDATensorizeContext ctx,
                                 CUDATensorizeParam tensorize_param,
                                 te::Operation consumer, te::Tensor inp,
                                 te::Tensor shared_attach_tensor,
                                 tir::IterVar shared_attach_axis,
                                 te::Tensor frag_attach_tensor,
                                 tir::IterVar frag_attach_axis, String scope,
                                 te::TensorIntrin intrinsic)
    {
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
      if (ctx->tz_used)
      {
        factors.push_back(tensorize_param->tz_size);
        tz_id = cur_id;
        cur_id += 1;
      }
      if (ctx->ty_used)
      {
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
      if (tz_id >= 0)
      {
        sch[inp_shared].bind(tiled[tz_id], te::thread_axis(Range(), "threadIdx.z"));
      }
      if (ty_id >= 0)
      {
        sch[inp_shared].bind(tiled[ty_id], te::thread_axis(Range(), "threadIdx.y"));
      }
      if (tx_id >= 0)
      {
        sch[inp_shared].bind(tiled[tx_id], te::thread_axis(Range(), "threadIdx.x"));
      }
      if (vec_id >= 0)
      {
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

    void ScheduleInterPath(te::Schedule sch, CUDATensorizeContext ctx,
                           CUDATensorizeParam tensorize_param)
    {
      te::Operation root_op = ctx->InterPathRootOp();
      sch[root_op].set_scope("shared");
      CHECK(ctx->path_attach_tensor.defined() && ctx->path_attach_axis.defined());
      sch[root_op].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
      for (auto op : ctx->InterPathNonRootOps())
      {
        CHECK(ctx->CanInline(op));
        sch[op].compute_inline();
      }
      // cooperative fetching
      Array<PrimExpr> factors;
      int cur_id{0}, tz_id{-1}, ty_id{-1}, tx_id{-1}, vec_id{-1};
      factors.push_back(-1);
      cur_id += 1;
      if (ctx->tz_used)
      {
        factors.push_back(tensorize_param->tz_size);
        tz_id = cur_id;
        cur_id += 1;
      }
      if (ctx->ty_used)
      {
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
      if (tz_id >= 0)
      {
        sch[root_op].bind(tiled[tz_id], te::thread_axis(Range(), "threadIdx.z"));
      }
      if (ty_id >= 0)
      {
        sch[root_op].bind(tiled[ty_id], te::thread_axis(Range(), "threadIdx.y"));
      }
      if (tx_id >= 0)
      {
        sch[root_op].bind(tiled[tx_id], te::thread_axis(Range(), "threadIdx.x"));
      }
      if (vec_id >= 0)
      {
        sch[root_op].vectorize(tiled[vec_id]);
      }
    }

    void ScheduleFirstOpLocality(te::Schedule sch, CUDATensorizeContext ctx,
                                 CUDATensorizeParam tensorize_param)
    {
      te::Operation first_op = ctx->state->first_op;
      CHECK(first_op->num_outputs() == 1)
          << "Only expect one output from first op.\n";
      te::Tensor first_out = first_op.output(0);
      te::Operation consumer;
      if (ctx->HasInterPath())
      {
        Array<te::Operation> path_ops = ctx->InterPathNonRootOps();
        if (path_ops.size() > 0U)
        {
          consumer = path_ops[0];
        }
        else
        {
          consumer = ctx->InterPathRootOp();
        }
      }
      else
      {
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
      for (int i = 0; i < (int)spatial_extents.size() - num_tensorize_iters; ++i)
      {
        if ((spatial_extents[i] >= tensorize_param->tz_size) && (ctx->tz_used) &&
            (tz_id < 0))
        {
          tz_id = i;
        }
        else if ((spatial_extents[i] >= tensorize_param->ty_size) &&
                 (ctx->ty_used) && (ty_id < 0))
        {
          ty_id = i;
        }
      }
      std::vector<tir::IterVar> outers(shared_cop->axis.size(), tir::IterVar()),
          inners(shared_cop->axis.size(), tir::IterVar());
      if (tz_id >= 0)
      {
        CHECK(tz_id < (int)shared_cop->axis.size());
        tir::IterVar axis = shared_cop->axis[tz_id];
        tir::IterVar outer, inner;
        sch[shared].split_by_nparts(axis, tensorize_param->tz_size, &outer, &inner);
        sch[shared].bind(outer, te::thread_axis(Range(), "threadIdx.z"));
        outers[tz_id] = outer;
        inners[tz_id] = inner;
      }
      if (ty_id >= 0)
      {
        CHECK(ty_id < (int)shared_cop->axis.size());
        tir::IterVar axis = shared_cop->axis[ty_id];
        tir::IterVar outer, inner;
        sch[shared].split_by_nparts(axis, tensorize_param->ty_size, &outer, &inner);
        sch[shared].bind(outer, te::thread_axis(Range(), "threadIdx.y"));
        outers[ty_id] = outer;
        inners[ty_id] = inner;
      }
      for (int i = 0; i < (int)shared_cop->axis.size() - num_tensorize_iters; ++i)
      {
        if ((i != tz_id) && (i != ty_id))
        {
          outers[i] = shared_cop->axis[i];
        }
      }
      // reorder
      Array<tir::IterVar> order;
      for (auto list : {outers, inners})
      {
        for (auto iv : list)
        {
          if (iv.defined())
          {
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
      for (auto list : {outers, inners})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            frag_attach_axis = list[i];
            break;
          }
        }
        if (frag_attach_axis.defined())
        {
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
      for (int i = 0; i < (int)first_cop->axis.size(); ++i)
      {
        spatial_index.push_back(i);
      }
      CHECK(ctx->ValidTensorizeFusion(spatial_index, tensorize_spatial_index))
          << "The fusion and tensorize decisions are not valid.\n";
      spatial_index.erase(spatial_index.begin() + (spatial_index.size() -
                                                   tensorize_spatial_index.size()),
                          spatial_index.end());
      // reduce
      tensorize_reduce_index = ctx->TensorizeReduceAxis(ctx->state->first_op);
      for (int i = 0; i < (int)first_cop->reduce_axis.size(); ++i)
      {
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
      for (int i = 0; i < (int)reduce_extents.size(); ++i)
      {
        if (reduce_extents[i] > split_extent)
        {
          split_extent = reduce_extents[i];
          split_id = i;
        }
      }
      std::vector<tir::IterVar> outs(reduce_extents.size(), tir::IterVar()),
          medians(reduce_extents.size(), tir::IterVar()),
          ins(reduce_extents.size(), tir::IterVar());
      if (split_id >= 0)
      {
        tir::IterVar axis = first_cop->reduce_axis[split_id];
        Array<tir::IterVar> tiled =
            ctx->Split(sch, first_op, axis,
                       {-1, tensorize_param->block_rx, tensorize_param->warp_rx});
        outs[split_id] = tiled[0];
        medians[split_id] = tiled[1];
        ins[split_id] = tiled[2];
      }
      for (int i = 0; i < (int)reduce_extents.size(); ++i)
      {
        if (i != split_id)
        {
          outs[i] = first_cop->reduce_axis[i];
        }
      }
      Array<tir::IterVar> new_order;
      for (auto list : {outs, medians, ins})
      {
        for (auto iv : list)
        {
          if (iv.defined())
          {
            new_order.push_back(iv);
          }
        }
      }
      for (auto ind : spatial_index)
      {
        new_order.push_back(first_cop->axis[ind]);
      }
      for (auto ind : tensorize_spatial_index)
      {
        new_order.push_back(first_cop->axis[ind]);
      }
      for (auto ind : tensorize_reduce_index)
      {
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
      for (auto list : {medians})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            prologue_frag_attach_axis = list[i];
            break;
          }
        }
        if (prologue_frag_attach_axis.defined())
        {
          break;
        }
      }
      CHECK(prologue_frag_attach_axis.defined())
          << "Can't find first_op's prologue fragment compute_at position during "
             "scheduling.\n";
      ctx->first_frag_attach_tensor = first_op.output(0);
      ctx->first_frag_attach_axis = prologue_frag_attach_axis;
      tir::IterVar prologue_shared_attach_axis;
      for (auto list : {outs})
      {
        for (int i = (int)list.size() - 1; i >= 0; --i)
        {
          if (list[i].defined())
          {
            prologue_shared_attach_axis = list[i];
            break;
          }
        }
        if (prologue_shared_attach_axis.defined())
        {
          break;
        }
      }
      CHECK(prologue_shared_attach_axis.defined())
          << "Can't find first_op's prologue shared compute_at position during "
             "scheduling.\n";
      ctx->first_prologue_shared_attach_tensor = first_op.output(0);
      ctx->first_prologue_shared_attach_axis = prologue_shared_attach_axis;
    }

    te::Schedule TensorizeCUDA(Layer layer, TensorizeHyperFusionState state,
                               hardware::HardwareParam cuda_param,
                               CUDATensorizeParam tensorize_param)
    {
      te::Schedule sch = te::create_schedule(layer->ops);
      CUDATensorizeContext ctx = CUDATensorizeContext(layer, state, cuda_param);
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
      for (auto inp : ctx->second_frag->op->InputTensors())
      {
        CHECK(ctx->state->tensorize_intrinsics.count(ctx->state->second_op));
        PackedIntrinsic pintrin =
            ctx->state->tensorize_intrinsics.at(ctx->state->second_op);
        CHECK(count_num_input < (int)pintrin->load_scopes.size());
        if ((inp->op == ctx->state->first_op) || ctx->IsInInterPath(inp->op))
        {
          // input from first op or inter path
          ScheduleInputFrag(sch, ctx, tensorize_param, ctx->second_frag->op, inp,
                            ctx->second_frag_attach_tensor,
                            ctx->second_frag_attach_axis,
                            pintrin->load_scopes[count_num_input],
                            pintrin->load_intrinsics[count_num_input]);
        }
        else
        {
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
      for (auto op_list : ctx->state->second_op_prologue)
      {
        for (auto op : op_list)
        {
          CHECK(ctx->CanInline(op));
          sch[op].compute_inline();
        }
      }
      /*
   * Schedule inter path
   */
      if (ctx->HasInterPath())
      {
        ScheduleInterPath(sch, ctx, tensorize_param);
      }
      /*
   * Schedule first op
   */
      ScheduleFirstOpLocality(sch, ctx, tensorize_param);
      /*
   * Schedule first op's inputs
   */
      count_num_input = 0;
      for (auto inp : ctx->state->first_op->InputTensors())
      {
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
      for (auto op_list : ctx->state->first_op_prologue)
      {
        for (auto op : op_list)
        {
          CHECK(ctx->CanInline(op));
          sch[op].compute_inline();
        }
      }
      return sch;
    }
    std::pair<cost_t, FusionItem> setTemplatesAndSearch(IterGraph ig,
                                                        SearchDriver searchDriver)
    {
      FusionSpace fusionSpace = searchDriver->getFusionSpace();
      Array<IntImm> firstOpPermute, firstOpTiling;
      size_t idx = 0;
      for (auto iv : ig->_firstOpIters)
      {
        firstOpPermute.push_back(IntImm(DataType::Int(32), idx++));
        firstOpTiling.push_back(IntImm(DataType::Int(32), 1));
      }
      fusionSpace->setFirstOpPermuteMandatory({firstOpPermute});
      fusionSpace->setFirstOpTilingMandatory(firstOpTiling);
      struct FusionTemplate
      {
        std::string name;
        std::string secondOpPermute[3];
        size_t attachPos;
      };

      // the F1~F7 templates
      FusionTemplate fusionTemplates[7] = {
          {"F1", {"S1", "S2", "R"}, 3}, {"F2", {"S1", "S2", "R"}, 2}, {"F3", {"S1", "R", "S2"}, 2}, {"F4", {"S2", "R", "S1"}, 2}, {"F5", {"S1", "S2", "R"}, 1}, {"F6", {"S2", "S1", "R"}, 1}, {"F7", {"R", "S1", "S2"}, 1}};

      std::vector<int> secondOpR, secondOpS1, secondOpS2;

      idx = 0;
      for (auto iv : ig->_secondOpIters)
      {
        if (iv->iv_type == IV_Type::FIRSTSPATIAL)
          secondOpS1.push_back(idx++);
        else if (iv->iv_type == IV_Type::SECONDSPATIAL)
          secondOpS2.push_back(idx++);
        else
          secondOpR.push_back(idx++);
      }
      std::unordered_map<std::string, std::vector<int>> str2Ivs = {
          {"S1", secondOpS1}, {"S2", secondOpS2}, {"R", secondOpR}};
      cost_t bestCost = INFINITY;
      FusionItem bestFusionItem;
      for (auto fusionTemplate : fusionTemplates)
      {
        size_t cacheLevel = 0;
        std::vector<int> secondOpPermute;
        int secondOpTilingFactors[ig->_secondOpIters.size()];
        int fusionAttachPos = 0;
        for (size_t i = 0; i < ig->_secondOpIters.size(); i++)
          secondOpTilingFactors[i] = 1;
        for (auto ivType : fusionTemplate.secondOpPermute)
        {
          secondOpPermute.insert(secondOpPermute.end(), str2Ivs[ivType].begin(),
                                 str2Ivs[ivType].end());
          if (cacheLevel < fusionTemplate.attachPos)
          {
            fusionAttachPos += str2Ivs[ivType].size();
            for (auto idx : str2Ivs[ivType])
            {
              secondOpTilingFactors[idx] = -1;
            }
          }
          ++cacheLevel;
        }
        Array<IntImm> secondOpPermute_;
        for (auto i : secondOpPermute)
        {
          secondOpPermute_.push_back(IntImm(DataType::Int(32), i));
        }
        fusionSpace->setSecondOpPermuteMandatory({secondOpPermute_});

        Array<IntImm> secondOpTilingFactors_;
        for (auto i : secondOpTilingFactors)
          secondOpTilingFactors_.push_back(IntImm(DataType::Int(32), i));

        fusionSpace->setAttacchMandatory(
            {IntImm(DataType::Int(32), fusionAttachPos)});
        fusionSpace->setSecondOpTilingMandatory(secondOpTilingFactors_);

        auto res = searchDriver->search_with_loss();
        cost_t curCost = res.first;
        FusionItem fusionItem = GetRef<FusionItem>(res.second.as<FusionItemNode>());
        if (!fusionItem.defined())
        {
          LOG(WARNING) << "search for " << fusionTemplate.name << " failed";
          continue;
        }
        LOG(INFO) << "End searching pattern " << fusionTemplate.name
                  << ", best cost is " << curCost << ", fusion Item is \n"
                  << fusionItem << std::endl;
        if (curCost < bestCost)
        {
          bestCost = curCost;
          bestFusionItem = fusionItem;
        }
      }
      CHECK(bestCost < 1e9) << "no valid fusion choice found";
      return {bestCost, bestFusionItem};
    }

    void setTemplatesAndSearchPrunedDFS(
        IterGraph ig, hardware::HardwareParam hw_param, int bytePerEle,
        std::string searchType = "normal", std::string mode = "survey",
        std::vector<FusionInfo> *data = NULL, cost_t *cost_p = NULL,
        FusionItem *fusionItem_p = NULL)
    {
      Array<IntImm> firstOpPermute, firstOpTiling;
      size_t idx = 0;
      for (auto iv : ig->_firstOpIters)
      {
        firstOpPermute.push_back(IntImm(DataType::Int(32), idx++));
        firstOpTiling.push_back(IntImm(DataType::Int(32), 1));
      }
      ig->setFirstOpPermute(firstOpPermute);
      ig->setFirstOpTiling(firstOpTiling);

      struct FusionTemplate
      {
        std::string name;
        std::string secondOpPermute[4];
        size_t attachPos;
      };

      // the F1~F7 templates
      // FusionTemplate fusionTemplates[7] = {
      //     {"F1", {"S1", "S2", "R"}, 3}, {"F2", {"S1", "S2", "R"}, 2}, {"F3",
      //     {"S1", "R", "S2"}, 2}, {"F4", {"S2", "R", "S1"}, 2}, {"F5", {"S1",
      //     "S2", "R"}, 1}, {"F6", {"S2", "S1", "R"}, 1}, {"F7", {"R", "S1", "S2"},
      //     1}};
      FusionTemplate fusionTemplates[1] = {{"F1", {"B", "S1", "S2", "R"}, 4}};

      std::vector<int> secondOpR, secondOpS1, secondOpS2;

      idx = 0;
      for (auto iv : ig->_secondOpIters)
      {
        if (iv->iv_type == IV_Type::FIRSTSPATIAL)
          secondOpS1.push_back(idx++);
        else if (iv->iv_type == IV_Type::SECONDSPATIAL)
          secondOpS2.push_back(idx++);
        else
          secondOpR.push_back(idx++);
      }
      
      if (verbose){
        fprintf(stdout, "search space: ");
        for (auto iv: ig->_secondOpIters){
          std::cout << iv << " ";
        }
        std::cout << std::endl;
      }

      std::unordered_map<std::string, std::vector<int>> str2Ivs = {
          {"S1", secondOpS1}, {"S2", secondOpS2}, {"R", secondOpR}};
      cost_t bestCost = INFINITY;
      FusionItem bestFusionItem;
      size_t nSecondOpIters = ig->_secondOpIters.size();
      int secondOpBounds[nSecondOpIters];
      for (size_t i = 0; i < nSecondOpIters; i++)
      {
        secondOpBounds[i] = ig->_secondOpIters[i]->ext;
      }
      std::vector<FusionInfo> candidates(MAX_CANDIDATES_NUMBER);
      size_t itemCnt = 0;
      ig->setConfig(hw_param, bytePerEle);
      std::vector<int> fusion_levels = ig->fusionLevels;
      if (searchType == "nofuse") fusion_levels = {fusion_levels.back()};
      for (auto fusionLevel : fusion_levels)
      {
        std::cout << "fusionLevel: " << fusionLevel << std::endl;
        int itemBeforeSearch = itemCnt;
        ig->setFusionLevel(fusionLevel);

        double bestLevelCost = INFINITY;
        int bestFactors[nSecondOpIters];
        for (auto fusionTemplate : fusionTemplates)
        {
          size_t cacheLevel = 0;
          std::vector<int> secondOpPermute;
          int secondOpTilingFactors[nSecondOpIters];
          std::vector<int> secondOpUnsetIndices;
          std::vector<int> secondOpOuterIndices;
          int fusionAttachPos = 0;
          for (size_t i = 0; i < nSecondOpIters; i++)
            secondOpTilingFactors[i] = 1;
          for (auto ivType : fusionTemplate.secondOpPermute)
          {
            secondOpPermute.insert(secondOpPermute.end(), str2Ivs[ivType].begin(),
                                   str2Ivs[ivType].end());
            if (cacheLevel < fusionTemplate.attachPos)
            {
              fusionAttachPos += str2Ivs[ivType].size();
              for (auto idx : str2Ivs[ivType])
              {
                secondOpTilingFactors[idx] = -1;
                secondOpUnsetIndices.push_back(idx);
                secondOpOuterIndices.push_back(ig->_secondOpIters[idx]->index);
              }
            }
            ++cacheLevel;
          }
          Array<IntImm> secondOpPermute_;
          for (auto i : secondOpPermute)
            secondOpPermute_.push_back(IntImm(DataType::Int(32), i));
          ig->setSecondOpPermute(secondOpPermute_);
          ig->setAttach(fusionAttachPos);

          auto getCost = [&secondOpTilingFactors, &ig,
                          &nSecondOpIters](double *occupancy = NULL, double *parallelism = NULL, double *memUse = NULL) {
            Array<IntImm> secondOpTilingFactors_;
            for (size_t i = 0; i < nSecondOpIters; i++)
              secondOpTilingFactors_.push_back(
                  IntImm(DataType::Int(32), secondOpTilingFactors[i]));
            ig->setSecondOpTiling(secondOpTilingFactors_);
            return ig->getCost(occupancy, parallelism, memUse);
          };

          std::function<void(int)> dfs = [&](size_t idx) {
            bool valid;
            double cost;
            if (idx == secondOpUnsetIndices.size())
            {
              double occupancy, parallelism, memUse;
              std::tie(valid, cost) = getCost(&occupancy, &parallelism, &memUse);
              FusionInfo fusionInfo;
              ig->getTotalDM(fusionInfo.features);
              fusionInfo.cacheOccupancy = occupancy;
              fusionInfo.cost = cost;
              fusionInfo.n_block = ig->getNumOfBlocks();
              fusionInfo.outerCost = ig->_outerCost;
              fusionInfo.maxThreadIter = ig->_maxThreadIter;
              fusionInfo.parallelism = parallelism;
              fusionInfo.secondOpOuterIndices = secondOpOuterIndices;
              fusionInfo.fusionLevel = fusionLevel;
              fusionInfo.computation =
                  ig->getFirstOpWorkload() + ig->getSecondOpWorkload();
              fusionInfo.lower_bounds_for_upper_cache_level 
                = fusionInfo.upper_bounds_for_lower_cache_level = ig->bounds;
              fusionInfo.valid = true;
              fusionInfo.boundsAfterParallel = ig->_boundsAfterParallel;
              fusionInfo.parallelFactor = ig->_parallelSchedule;
              for (auto idx : secondOpUnsetIndices)
                fusionInfo.secondOpOuterTilingFactors.push_back(
                    secondOpTilingFactors[idx]);
              fusionInfo.memUse = memUse;
              if (verbose){
                std::cout << "[search]: ";
                for (size_t i = 0; i < secondOpUnsetIndices.size(); i++)
                {
                  tir::Var var = ig->_secondOpIters[secondOpUnsetIndices[i]]->name;
                  int factor = secondOpTilingFactors[i];
                  std::cout << "(" << var << ": " << factor << "), ";
                }
                std::cout << " : " << valid << " " << cost << std::endl;
              }
              if (valid)
              {
                if (cost < bestLevelCost)
                {
                  bestLevelCost = cost;
                  memcpy(bestFactors, secondOpTilingFactors,
                         sizeof(int) * nSecondOpIters);
                  if (mode == "best" && itemCnt < MAX_CANDIDATES_NUMBER && (itemCnt - itemBeforeSearch) < 0.75 * MAX_CANDIDATES_NUMBER)
                    candidates[itemCnt++] = fusionInfo;
                }
                if (mode != "best" && itemCnt < MAX_CANDIDATES_NUMBER && (itemCnt - itemBeforeSearch) < 0.75 * MAX_CANDIDATES_NUMBER)
                {
                  candidates[itemCnt++] = fusionInfo;
                }
              }
              return;
            }
            for (size_t i = idx; i < secondOpUnsetIndices.size(); i++)
              secondOpTilingFactors[secondOpUnsetIndices[i]] = 1;
            std::tie(valid, cost) = getCost();
            if (!valid)
              return;
            for (size_t i = idx; i < secondOpUnsetIndices.size(); i++)
            {
              secondOpTilingFactors[secondOpUnsetIndices[i]] =
                  secondOpBounds[secondOpUnsetIndices[i]];
            }
            if (mode == "best" || mode == "test")
            {
              std::tie(valid, cost) = getCost();
              if (cost > bestLevelCost)
                return;
            }
            int pos = secondOpUnsetIndices[idx];
            int bound = secondOpBounds[pos];
            for (int i = 1; i <= bound; i++)
            {
              secondOpTilingFactors[pos] = i;
              if (searchType == "stochastic")
                secondOpTilingFactors[pos] = (random() % bound) + 1;
              dfs(idx + 1);
            }
          };
          // LOG(INFO) << "Begin searching pattern " << fusionTemplate.name;
          dfs(0);
          if (bestLevelCost < bestCost)
          {
            bestCost = bestLevelCost;
            Array<IntImm> secondOpTiling_;
            for (size_t i = 0; i < nSecondOpIters; i++)
              secondOpTiling_.push_back(IntImm(DataType::Int(32), bestFactors[i]));
            bestFusionItem =
                buildFusionItem(firstOpTiling, secondOpTiling_, firstOpPermute,
                                secondOpPermute_, fusionAttachPos, fusionLevel);
          }
        }
        std::cout << "candidates for fusionLevel " << fusionLevel << ": "
                  << itemCnt - itemBeforeSearch << std::endl;
      }
      CHECK(itemCnt <= MAX_CANDIDATES_NUMBER);
      candidates.resize(itemCnt);
      std::sort(candidates.begin(), candidates.end(),
                [](FusionInfo &first, FusionInfo &second) {
                  return first.cost < second.cost;
                });

      std::cout << "candidates.size(): " << candidates.size() << std::endl;
      if (verbose) 
        std::cout << "best candidate: \n" << candidates.front() << std::endl;

      if (data)
      {
        if (mode == "best")
        {
          std::vector<FusionInfo> candidates_;
          std::unordered_map<int, int> candidateSize;
          for (auto fusionLevel : ig->fusionLevels)
          {
            candidateSize[fusionLevel] = 0;
          }
          for (auto candidate : candidates)
          {
            candidateSize[candidate.fusionLevel] += 1;
            if (candidateSize[candidate.fusionLevel] > SELECT_TOP)
              continue;
            candidates_.push_back(candidate);
          }
          *data = {candidates_};
        }
        else if (mode == "test")
        {
          *data = {candidates[0]};
        }
        else if (mode == "survey")
        {
          std::vector<FusionInfo> candidates_;
          for (size_t i = 0; i < SURVEY_CANDIDATES_NUMBER; i++){
            candidates_.push_back(candidates[random() % candidates.size()]);
          }
          if (verbose){
            std::function<std::pair<double, double>(const std::vector<FusionInfo>&)> get_stats = 
            [](const std::vector<FusionInfo>& vec){
              double ave = 0, std = 0;
              for(auto & item: vec) 
                ave += item.cost, std += item.cost * item.cost;
              ave /= vec.size();
              std /= vec.size();
              std = sqrt(std - ave * ave);
              return std::make_pair(ave, std);
            };
            double ave, std;
            std::tie(ave, std) = get_stats(candidates);
            std::cout << "original data: ave " << ave << ", std " << std << "; ";
            std::tie(ave, std) = get_stats(candidates_);
            std::cout << "sampled data: ave " << ave << ", std " << std << std::endl; 
          }

          *data = move(candidates_);
        }
      }
      CHECK(bestFusionItem.defined());
      if (cost_p)
        *cost_p = bestCost;
      if (fusionItem_p)
        *fusionItem_p = bestFusionItem;
      CHECK(bestCost != INFINITY) << "no valid fusion choice found";
      return;
    }
    /*! build the fusion choice*/
    FusionChoice buildFusionChoice(SerialFusionState sfs,
                                   hardware::HardwareParam hw_param, String dtype,
                                   int simple_mode)
    {
      // CHECK(sfs->tensorizeAxes.defined());
      IterGraph ig = buildIterGraph(sfs);
      SearchDriver searchDriver =
          buildSearchDriver(ig, {"static analysis"}, "bruteForce", hw_param, dtype);

      cost_t bestCost = 0;
      FusionItem bestFusionItem = {};
      std::unordered_map<std::string, int32_t> m = {
          {"float32", 4}, {"float64", 8}, {"float16", 2}, {"int16", 2}, {"int32", 4}, {"int64", 8}};
      CHECK(m.count(dtype));
      int bytePerEle = m[dtype];
      if (simple_mode == 1)
      {
        ig->setConfig(hw_param, bytePerEle);
        auto vec2array = [](std::vector<int> v) {
          Array<IntImm> ret;
          for (auto v_ : v)
            ret.push_back(IntImm(DataType::Int(32), v_));
          return ret;
        };
        std::vector<int> simpleTiling, simplePermute;
        for (size_t i = 0; i < ig->_firstOpIters.size(); i++)
        {
          simpleTiling.push_back(1);
          simplePermute.push_back(i);
        }

        bestFusionItem =
            buildFusionItem(vec2array(simpleTiling), vec2array(simpleTiling),
                            vec2array(simplePermute), vec2array(simplePermute), 1, ig->fusionLevels[0]);
      }
      else if (simple_mode == -1)
      {
        setTemplatesAndSearchPrunedDFS(ig, hw_param, m[dtype], "normal", "best",
                                         NULL, &bestCost, &bestFusionItem);
      }
      else if (simple_mode == 0)
      {
        std::tie(bestCost, bestFusionItem) =
            setTemplatesAndSearch(ig, searchDriver);
      }
      // std::cout << "best fusionItem: \n" << bestFusionItem;
      // ig->setFusion(bestFusionItem);
      // std::cout << "best candidate, cost: " << bestCost << std::endl;
      // ig->visualize();

      FusionResult fusionResult = GetRef<FusionResult>(
          searchDriver->eval(bestFusionItem)[0].as<FusionResultNode>());
      CHECK(fusionResult.defined());
      Array<te::IterVar> secondOpIters;
      Array<te::IterVar> remainedIters;
      Array<IntImm> secondOpOuterIndices;
      Array<IntImm> secondOpOuterSplitFactors;
      const te::ComputeOpNode *op2 = ig->op2.as<te::ComputeOpNode>();
      PermuteItem permute = bestFusionItem->secondOpPermute;
      size_t idx = 0;
      size_t attachPos = bestFusionItem->attachPos->attachPos;
      for (auto i_ : permute->permute)
      {
        size_t i = ig->_secondOpIters[i_->value]->index;
        if (i >= op2->axis.size())
        {
          // reduce axis
          i -= op2->axis.size();
          secondOpIters.push_back(op2->reduce_axis[i]);
        }
        else
        {
          // spatial axis
          secondOpIters.push_back(op2->axis[i]);
        }
        if (idx < attachPos)
        {
          secondOpOuterSplitFactors.push_back(
              bestFusionItem->secondOpTiling->factors[i_->value]);
          secondOpOuterIndices.push_back(
              IntImm(DataType::Int(32), ig->_secondOpIters[i_->value]->index));
        }
        idx++;
      }
      for (auto iv : ig->secondOpTensorizeIters)
      {
        size_t i = iv->index;
        if (i >= op2->axis.size())
        {
          // reduce axis
          i -= op2->axis.size();
          secondOpIters.push_back(op2->reduce_axis[i]);
        }
        else
        {
          // spatial axis
          secondOpIters.push_back(op2->axis[i]);
        }
      }
      CHECK(secondOpIters.size() == (op2->axis.size() + op2->reduce_axis.size()))
          << "the fusion axes number does not match";
      return FusionChoice(ig->op1, ig->op2, secondOpIters, attachPos - 1,
                          secondOpOuterSplitFactors, secondOpOuterIndices,
                          bestFusionItem, fusionResult);
    }

    FusionContext::FusionContext(SerialFusionState sfs,
                                 std::vector<CPUTensorizeParam> schParams,
                                 Layer layer, TensorizeHyperFusionState state, String path,
                                 hardware::HardwareParam hw_param, int bytePerEle)
    {
      auto node = make_object<FusionContextNode>();
      node->sfs = sfs;
      node->layer = layer;
      node->state = state;
      node->hw_param = hw_param;
      node->bytePerEle = bytePerEle;
      node->schParams = schParams;
      node->size = schParams.size();
      if (node->size <= 5)
        LOG(WARNING) << "only " << node->size << " available candidates";
      node->path = path;
      data_ = node;
    }
    Map<String, FloatImm> FusionContextNode::getFeature(int i){
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      Map<String, FloatImm> ret;
      for (auto item: schParams[i]->fusionInfo.features){
        String k = String(item.first);
        FloatImm v = FloatImm(DataType::Float(64), item.second);
        ret.Set(k, v);
      }
      return ret;
    }
    int FusionContextNode::getFusionLevel(int i){
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      return schParams[i]->fusionInfo.fusionLevel;
    }
    double FusionContextNode::getPredCost(int i)
    {
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      double cost = 0;
      for (auto _ : schParams[i]->costs)
        cost += _;
      return cost;
    }
    double FusionContextNode::getComputation(int i)
    {
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      return schParams[i]->fusionInfo.computation;
    }
    Array<FloatImm> FusionContextNode::getOccupancy(int i)
    {
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      return schParams[i]->cacheOccupancy;
    }
    Array<FloatImm> FusionContextNode::getPredCostList(int i)
    {
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      const CPUTensorizeParam &param = schParams[i];
      Array<FloatImm> ret;
      CHECK(param->firstOpCosts.size() == param->secondOpCosts.size());
      for (size_t i = 0; i < param->firstOpCosts.size(); i++)
      {
        double tmp = param->firstOpCosts[i] + param->secondOpCosts[i];
        ret.push_back(FloatImm(DataType::Float(32), tmp));
      }
      for (size_t i = 0; i < param->commCosts.size(); i++)
      {
        double tmp = param->commCosts[i];
        if (i == 0)
          tmp += param->fusionInfo.cost;
        ret.push_back(FloatImm(DataType::Float(32), tmp));
      }
      return ret;
    }
    std::ostream &operator<<(std::ostream &o, const FusionInfo &fusionInfo)
    {
      /*
      struct FusionInfo {
        bool valid;
        std::vector<int> secondOpOuterIndices;
        std::vector<int> secondOpOuterTilingFactors;
        int fusionLevel;
        int n_block;
        int parallelism;
        double cost;
        double computation;
        Map<tir::Var, IntImm> bounds;
        Map<tir::Var, IntImm> boundsAfterParallel;
        std::unordered_map<int, IntImm> parallelFactor;
        double cacheOccupancy;
        double memUse;
        double outerCost;
        double maxThreadIter;
        std::unordered_map<std::string, double> features;
      };*/
      o << "===========fusion_info===========" << std::endl;
      o << "lower_bounds_for_upper_cache_level: " << fusionInfo.lower_bounds_for_upper_cache_level << std::endl;
      o << "upper_bounds_for_lower_cache_level: " << fusionInfo.upper_bounds_for_lower_cache_level << std::endl;
      o << "bounds after parallel" << fusionInfo.boundsAfterParallel << std::endl;
      std::cout << "parallelFactor: ";
      for (auto idxfactor : fusionInfo.parallelFactor)
        o << "(" << idxfactor.first << ", " << idxfactor.second << "),";
      std::cout << std::endl;
      o << "occupancy: " << fusionInfo.cacheOccupancy << std::endl;
      o << "fusionLevel: " << fusionInfo.fusionLevel << std::endl;
      o << "parallelism: " << fusionInfo.parallelism << std::endl;
      o << "fusion cost: " << fusionInfo.cost << std::endl;
      o << "outerIndices.size(): " << fusionInfo.secondOpOuterIndices.size()
        << std::endl;
      o << "#block: " << fusionInfo.n_block << std::endl;
      o << "outer cost" << fusionInfo.outerCost << std::endl;
      o << "max iter: " << fusionInfo.maxThreadIter << std::endl;
      o << "==================================" << std::endl;
      return o;
    }
    FusionContext buildFusionContext(SerialFusionState sfs, Layer layer,
                                     TensorizeHyperFusionState state,
                                     String path, hardware::HardwareParam hw_param,
                                     String searchType = "stochastic",
                                     String mode = "survey",
                                     String dtype = "float32")
    {
      std::vector<FusionInfo> fusionInfo(SURVEY_CANDIDATES_NUMBER);
      for (auto it : fusionInfo)
        it.valid = false;
      std::vector<CPUTensorizeParam> schParams;
      // CHECK(sfs->tensorizeAxes.defined());
      IterGraph ig = buildIterGraph(sfs);
      if (verbose){
        std:: cout << "iterGraph: " << std::endl;
        std::cout << ig << std::endl;
      }
      std::unordered_map<std::string, int32_t> m = {{"float32", 4}, {"float64", 8}, {"float16", 2}, {"int16", 2}, {"int32", 4}, {"int64", 8}};
      CHECK(m.count(dtype));
      int bytePerEle = m[dtype];
      if (searchType == "rule_based"){
        setTemplatesAndSearchByRule(
          ig, hw_param, bytePerEle, &fusionInfo, true
        );
      }
      else {
        setTemplatesAndSearchPrunedDFS(ig, hw_param, bytePerEle, searchType, mode,
                                       &fusionInfo);
      }
      OpHyperState op1, op2;
      std::tie(op1, op2) = sfs->getCubicOpPair();
      auto share_axes = share_axis_analysis(op1->op, op2->op);
      std::cout << "share_axes: " << share_axes << std::endl;
      size_t cnt = 0;
      int invalid_cnt = 0;
      for (auto& info : fusionInfo)
      {
        if (!info.valid)
          break;
        cnt += 1;
        if(verbose) {
          std::cout << "begin build Tensorize param: " << cnt << "/"
                    << fusionInfo.size() << std::endl;
          std::cout << info << std::endl;
        }
          
        CPUTensorizeParam schParam =
            buildCPUTensorizeParam(sfs, hw_param, bytePerEle, info, mode, searchType);
        printf("[buildFusionContext]: build Tensorize param %d\n", schParam->valid);
        if (!schParam->valid){
          invalid_cnt ++;
          continue; 
        }
        std::vector<double> costs;

        costs.insert(costs.end(), schParam->firstOpCosts.begin(),
                     schParam->firstOpCosts.end());
        costs.insert(costs.end(), schParam->secondOpCosts.begin(),
                     schParam->secondOpCosts.end());
        costs.push_back(info.cost);
        costs.insert(costs.end(), schParam->commCosts.begin(),
                     schParam->commCosts.end());

        schParam->costs = costs;
        double cost = 0;
        for (auto cost_ : costs)
          cost += cost_;
        schParam->cost = cost;
        schParams.push_back(schParam);
      }
      fprintf(stdout, "[buildFusionContext]: schParams.size() %d\n", schParams.size());
      std::sort(schParams.begin(), schParams.end(),
                [](CPUTensorizeParam &first, CPUTensorizeParam &second) {
                  if (first->valid && !second->valid)
                    return true;
                  if (!first->valid && second->valid)
                    return false;
                  if (!first->valid && !second->valid)
                    return true;
                  return first->cost < second->cost;
                });
      while (!(*schParams.rbegin())->valid)
        schParams.pop_back();
      if (verbose){
        std::cout << "[buildFusionContext]: valid: " << schParams.size() << "; invalid: " << invalid_cnt << std::endl;
      }
      return FusionContext(sfs, schParams, layer, state, path, hw_param,
                           m[dtype]);
    }

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getPredCost")
        .set_body_method<FusionContext>(&FusionContextNode::getPredCost);

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getComputation")
        .set_body_method<FusionContext>(&FusionContextNode::getComputation);

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getOccupancy")
        .set_body_method<FusionContext>(&FusionContextNode::getOccupancy);

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getPredCostList")
        .set_body_method<FusionContext>(&FusionContextNode::getPredCostList);
    
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getFeature")
        .set_body_method<FusionContext>(&FusionContextNode::getFeature);

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getFusionLevel")
        .set_body_method<FusionContext>(&FusionContextNode::getFusionLevel);

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildFusionContext")
        .set_body_typed([](SerialFusionState sfs, Layer layer,
                           TensorizeHyperFusionState state,
                           String data_path, hardware::HardwareParam hw_param,
                           String searchType, String mode, String dtype) {
          return buildFusionContext(sfs, layer, state, data_path, hw_param,
                                    searchType, mode, dtype);
        });
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.FusionChoice")
        .set_body_typed([](te::Operation first_op, te::Operation second_op,
                           Array<tir::IterVar> ordered_iters, int attach_pos) {
          return FusionChoice(first_op, second_op, ordered_iters, attach_pos);
        });

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.MatchInfo")
        .set_body_typed([](Array<tir::IterVar> axis, PackedIntrinsic intrin, String impl) {
          tir::StringImm impl_ = impl;
          return MatchInfo(axis, intrin, impl_);
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
                           int warp_ry, int warp_rz, int unroll_steps) {
          return CUDATensorizeParam(warp_size, ty_size, tz_size, input_vector_len,
                                    serial_y, serial_z, block_rx, block_ry,
                                    block_rz, warp_rx, warp_ry, warp_rz,
                                    unroll_steps);
        });

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeCUDA")
        .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                           hardware::HardwareParam cuda_param,
                           CUDATensorizeParam tensorize_param) {
          return TensorizeCUDA(layer, state, cuda_param, tensorize_param);
        });
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildFusionChoice")
        .set_body_typed([](SerialFusionState sfs, hardware::HardwareParam hw_param,
                           String dtype, int simple_mode) {
          return buildFusionChoice(sfs, hw_param, dtype, simple_mode);
        });
  } // namespace auto_tensorize

} // namespace ditto