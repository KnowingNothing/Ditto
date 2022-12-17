#include <auto_compute/graph.h>
#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/searchDriver.h>
#include <auto_tensorize/hyper_fusion.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/state.h>
#include <stack>
#include <tvm/driver/driver_api.h>
#include <tvm/te/schedule_pass.h>
#include <utils/iter_domain.h>
#include <iomanip>
namespace ditto
{

  namespace auto_tensorize
  {

    TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);
    TVM_REGISTER_NODE_TYPE(CPUTensorizeContextNode);
    TVM_REGISTER_NODE_TYPE(CPUTensorizeParamNode);
    TVM_REGISTER_NODE_TYPE(FusionContextNode);

    bool CPUTensorizeContextNode::HasEpilogue()
    {
      return (state->epilogue.size() > 0U);
    }

    te::Operation CPUTensorizeContextNode::EpilogueRootOp()
    {
      CHECK(this->HasEpilogue());
      return this->state->epilogue[(int)this->state->epilogue.size() - 1];
    }

    Array<te::Operation> CPUTensorizeContextNode::EpilogueNonRootOps()
    {
      Array<te::Operation> ret;
      for (int i = 0; i < (int)this->state->epilogue.size() - 1; ++i)
      {
        ret.push_back(this->state->epilogue[i]);
      }
      return ret;
    }

    bool CPUTensorizeContextNode::HasInterPath()
    {
      return (state->inter_path.size() > 0U);
    }

    te::Operation CPUTensorizeContextNode::InterPathRootOp()
    {
      // CHECK(this->HasInterPath());
      if (!this->HasInterPath())
      {
        return te::Operation();
      }
      return this->state->inter_path[(int)this->state->inter_path.size() - 1];
    }

    Array<te::Operation> CPUTensorizeContextNode::InterPathNonRootOps()
    {
      Array<te::Operation> ret;
      for (int i = 0; i < (int)this->state->inter_path.size() - 1; ++i)
      {
        ret.push_back(this->state->inter_path[i]);
      }
      return ret;
    }

    tir::IterVar CPUTensorizeContextNode::FuseAll(te::Schedule sch,
                                                  te::Operation op)
    {
      te::Operation sop = sch[op]->op;
      const te::ComputeOpNode *cop = sop.as<te::ComputeOpNode>();
      Array<tir::IterVar> axis = cop->axis;
      tir::IterVar fused;
      sch[op].fuse(axis, &fused);
      return fused;
    }

    void CPUTensorizeContextNode::Inline(te::Schedule sch, te::Operation op)
    {
      sch[op].compute_inline();
    }

    bool CPUTensorizeContextNode::CanInline(te::Operation op)
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

    bool CPUTensorizeContextNode::isBatchLikeDim(const te::Operation &op,
                                                 const tir::IterVar iv)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      CHECK(cop->body.size() == 1) << "Only expect one body.\n";
      IsBatchLikeDim checker;
      std::vector<int> ret;
      return checker.is_batch(cop->body[0], iv->var);
    }

    bool CPUTensorizeContextNode::isSpatial(te::Operation op, tir::IterVar iv)
    {
      const te::ComputeOpNode *cur_op = op.as<te::ComputeOpNode>();
      CHECK(cur_op);
      for (auto iv_ : cur_op->axis)
        if (iv.same_as(iv_))
          return true;
      return false;
    }

    CPUTensorizeContext::CPUTensorizeContext(Layer layer,
                                             TensorizeHyperFusionState state)
    {
      auto node = make_object<CPUTensorizeContextNode>();
      node->layer = layer;
      node->state = state;
      data_ = node;
    }

    /*!
     * \brief Get the tensorize iters
     */
    Array<tir::IterVar>
    CPUTensorizeContextNode::GetTensorizeIters(const te::Operation &op)
    {
      CHECK(this->state->tensorize_iters.count(op));
      Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
      return iters;
    }

    /*!
     * \brief Get the tensorize iters
     */
    Array<te::IterVar>
    CPUTensorizeContextNode::GetAllIters(const te::Operation &op)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      Array<te::IterVar> ret;
      for (auto iv : cop->axis)
        ret.push_back(iv);
      for (auto iv : cop->reduce_axis)
        ret.push_back(iv);
      return ret;
    }
    /*!
     * \brief Get the outer iter of the second op
     */
    std::pair<std::vector<int>, Array<IntImm>>
    CPUTensorizeContextNode::GetSecondOpOuterIndexAndSplitFactor()
    {
      std::vector<int> indices;
      for (auto i : state->secondOpOuterIndices)
      {
        indices.push_back(i->value);
      }
      return {indices, state->secondOpOuterTileFactors};
    }
    /*!
     * \brief if the axis of op is spatial
     */
    std::pair<Array<tir::IterVar>, Array<tir::IterVar>>
    CPUTensorizeContextNode::splitSpatialWithReduce(const te::Operation op,
                                                    Array<tir::IterVar> ivs)
    {
      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      CHECK(cop);
      Array<tir::IterVar> Spatial, Reduce;
      for (auto iv : ivs)
      {
        bool isSpatial = false;
        for (auto iv_s : cop->axis)
          if (iv == iv_s)
          {
            Spatial.push_back(iv);
            isSpatial = true;
            break;
          }
        if (!isSpatial)
          Reduce.push_back(iv);
      }
      return {Spatial, Reduce};
    }
    CPUTensorizeParam::CPUTensorizeParam(
        bool valid)
    {
      auto node = make_object<CPUTensorizeParamNode>();
      CHECK(valid == false);
      node->valid = valid;
      data_ = node;
    }
    CPUTensorizeParam::CPUTensorizeParam(
        OpHyperState op1, OpHyperState op2, int parallelism,
        std::vector<std::vector<int>> firstOpLoopOrder,
        std::vector<std::vector<int>> secondOpLoopOrder,
        std::vector<std::vector<int>> commonLoopOrder,
        std::unordered_map<int, Array<IntImm>> firstOpTilingFactor,
        std::unordered_map<int, Array<IntImm>> secondOpTilingFactor,
        std::unordered_map<int, Array<IntImm>> commonTilingFactor,
        FusionInfo fusionInfo, std::vector<double> firstOpCosts,
        std::vector<double> secondOpCosts, std::vector<double> commCosts,
        Array<FloatImm> cacheOccupancy,
        std::unordered_map<std::string, double> log)
    {
      auto node = make_object<CPUTensorizeParamNode>();
      node->op1 = op1;
      node->op2 = op2;
      node->parallelism = parallelism;
      node->firstOpLoopOrder = firstOpLoopOrder;
      node->secondOpLoopOrder = secondOpLoopOrder;
      node->firstOpTilingFactor = firstOpTilingFactor;
      node->secondOpTilingFactor = secondOpTilingFactor;
      node->commonLoopOrder = commonLoopOrder;
      node->commonTilingFactor = commonTilingFactor;
      node->fusionInfo = fusionInfo;
      node->firstOpCosts = firstOpCosts;
      node->secondOpCosts = secondOpCosts;
      node->commCosts = commCosts, node->cacheOccupancy = cacheOccupancy;
      node->log = log;
      node->valid = true;
      data_ = node;
    }

    tir::IterVar getAttachAxis(te::Schedule sch, te::Operation consumer, te::Operation producer, tir::IterVar begin, Array<tir::IterVar> ivs)
    {
      Array<tir::Var> producer_vars = utils::GetAccessVars(consumer, producer);
      std::unordered_map<tir::IterVar, bool> isProducerIV;
      for (auto iv : sch[consumer]->all_iter_vars)
      {
        isProducerIV[iv] = false;
        for (auto var : producer_vars)
        {
          if (var.same_as(iv->var))
          {
            isProducerIV[iv] = true;
            break;
          }
        }
      }
      for (auto rel : sch[consumer]->relations)
      {
        if (const te::SplitNode *s = rel.as<te::SplitNode>())
        {
          isProducerIV[s->outer] = isProducerIV[s->inner] = isProducerIV[s->parent];
        }
        else if (const te::FuseNode *s = rel.as<te::FuseNode>())
        {
          isProducerIV[s->fused] = (isProducerIV[s->inner] && isProducerIV[s->outer]);
        }
      }
      tir::IterVar last = begin;
      // std::cout << "begin: " << begin << std::endl;
      // for (auto iv: sch[consumer]->leaf_iter_vars){
      //   std::cout << "iv: " << iv << " " << isProducerIV[iv] << std::endl;
      // }
      for (auto iv : ivs)
      {
        if (!(iv->dom.defined() && iv->dom->extent.as<te::IntImmNode>() && iv->dom->extent.as<te::IntImmNode>()->value == 1) && !isProducerIV[iv])
        {
          break;
        }
        last = iv;
      }
      // std::cout << "end get attach axis" << std::endl;
      return last;
    }
    ScheduleContext::ScheduleContext(std::vector<CostAndFactor> data,
                                     int parallelism)
    {
      auto node = make_object<ScheduleContextNode>();
      node->data = data;
      node->size = data.size();
      node->parallelism = parallelism;
      data_ = node;
    }
    TVM_REGISTER_NODE_TYPE(ScheduleContextNode);

    void ScheduleEpilogue(te::Schedule sch, CPUTensorizeContext ctx,
                          CPUTensorizeParam tensorize_param)
    {
      te::Operation cur_op;
      if (ctx->HasEpilogue())
      {
        cur_op = ctx->EpilogueRootOp();
        auto iv = ctx->FuseAll(sch, cur_op);
        tir::IterVar outer, inner;
        sch[cur_op].split_by_nparts(iv, tensorize_param->parallelism, &outer,
                                    &inner);
        sch[cur_op].parallel(outer);
        Array<te::Operation> remain_ops = ctx->EpilogueNonRootOps();
        // the remaining non-root ops should be inlined
        for (auto op : remain_ops)
        {
          CHECK(ctx->CanInline(op));
          sch[op].compute_inline();
        }
      }
    }

    void SmartParallelSchedule(te::Stage s, CPUTensorizeParam tensorize_param)
    {
      // parallel on axis i if its access pattern f(..., i + 1, ...) - f(..., i, ...) >= minParallelstride
      // in both the intensor and the out tensor
      te::Operation op = s->op;
      if (!(op->InputTensors().size() == 1))
        return;
      if (!op.as<te::ComputeOpNode>())
        return;
      te::Operation child = op->InputTensors()[0]->op;
      Array<PrimExpr> op_access_indices;
      Array<Array<PrimExpr>> access_indices = utils::GetAccessIndices(op, child);

      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      if (cop->reduce_axis.size())
        return;
      for (auto iv : cop->axis)
      {
        op_access_indices.push_back(iv->var);
      }
      access_indices.push_back(op_access_indices);
      Map<tir::Var, IntImm> map;
      Array<tir::Var> vars;
      Map<tir::Var, tir::IterVar> var2iv;
      for (auto iv : cop->axis)
      {
        CHECK(iv->dom.defined() && iv->dom->extent.defined());
        map.Set(iv->var, runtime::Downcast<IntImm>(iv->dom->extent));
        vars.push_back(iv->var);
        var2iv.Set(iv->var, iv);
      }

      std::function<std::vector<size_t>(Array<PrimExpr>, Map<tir::Var, IntImm>)> inferhelper = [&](Array<PrimExpr> exprs, Map<tir::Var, IntImm> map)
      {
        std::vector<size_t> ret;
        Map<tir::Var, Range> ranges;
        for (auto kv : map)
        {
          ranges.Set(kv.first, Range(0, kv.second));
        }
        Map<tir::Var, PrimExpr> vars_to_infer;
        Array<tir::Var> vars;
        for (auto expr : exprs)
        {
          auto var = tir::Var();
          vars_to_infer.Set(var, expr);
          vars.push_back(var);
        }
        Map<tir::Var, Range> var_range_map = utils::InferRange(vars_to_infer, ranges);
        for (auto var : vars)
        {
          auto range = var_range_map.at(var);
          ret.push_back(range->extent.as<IntImmNode>()->value);
        }
        return ret;
      };

      std::vector<std::vector<size_t>> strides;
      for (auto access_index : access_indices)
      {
        strides.push_back(inferhelper(access_index, map));
      }

      auto helper = [](std::vector<size_t> v, std::string name)
      {
        std::cout << name << ": ";
        for (auto v__ : v)
          std::cout << v__ << " ";
        std::cout << std::endl;
      };

      std::function<size_t(std::vector<size_t>, Array<PrimExpr>, Map<tir::Var, IntImm>, Map<tir::Var, IntImm>)> poshelper =
          [&inferhelper, &helper](std::vector<size_t> stride, Array<PrimExpr> access_indices, Map<tir::Var, IntImm> map1, Map<tir::Var, IntImm> map2)
      {
        int ret = 0;
        auto indices1 = inferhelper(access_indices, map1);
        auto indices2 = inferhelper(access_indices, map2);
        CHECK(stride.size() == indices1.size());
        size_t acc = 1;
        for (size_t i = stride.size(); i != 0; i--)
        {
          int idx = i - 1;
          ret += (indices1[idx] - indices2[idx]) * acc;
          acc *= stride[idx];
        }

        return ret < 0 ? -ret : ret;
      };

      Map<tir::Var, IntImm> allzero;
      for (auto var : vars)
      {
        allzero.Set(var, IntImm(DataType::Int(32), 1));
      }
      const int minParallelStride = 64;
      Array<tir::IterVar> paralleliters, otheriters;
      for (auto var : vars)
      {
        std::cout << var << std::endl;
        Map<tir::Var, IntImm> onehot;
        for (auto tmpvar : vars)
        {
          if (tmpvar.same_as(var))
            onehot.Set(var, IntImm(DataType::Int(32), 2));
          else
            onehot.Set(tmpvar, IntImm(DataType::Int(32), 1));
        }
        size_t diff = 1e9;
        for (size_t i = 0; i < access_indices.size(); i++)
        {
          size_t posDiff = poshelper(strides[i], access_indices[i], onehot, allzero);
          diff = std::min(diff, posDiff);
        }
        if (diff >= minParallelStride)
        {
          paralleliters.push_back(var2iv[var]);
        }
        else
          otheriters.push_back(var2iv[var]);
      }
      Array<tir::IterVar> looporder;
      for (auto arr : {paralleliters, otheriters})
      {
        for (auto iv : arr)
        {
          looporder.push_back(iv);
        }
      }

      s.reorder(looporder);

      tir::IterVar fused, outer, inner;

      s.fuse(paralleliters, &fused);

      s.split_by_nparts(fused, tensorize_param->parallelism, &outer, &inner);

      s.parallel(outer);

      return;
    }

    void ScheduleFirstOpPrologue(te::Schedule sch, CPUTensorizeContext ctx,
                                 CPUTensorizeParam tensorize_param)
    {
      for (auto op_list : ctx->state->first_op_prologue)
      {
        bool isFirst = true;
        for (auto op : op_list)
        {
          if (isFirst)
          {
            // tir::IterVar attach = getAttachAxis(sch, ctx->state->first_op, op, {}, sch[ctx->state->first_op]->leaf_iter_vars);
            // // std::cout << "firstOpPrologue attach: " << attach << std::endl;
            // if (attach.defined()){
            //   sch[op].compute_at(sch[ctx->state->first_op], attach);
            // }
            // else{
            //   SimpleSchedule(sch[op], tensorize_param);
            // }
            SmartParallelSchedule(sch[op], tensorize_param);
            isFirst = false;
          }
          else
          {
            sch[op].compute_inline();
          }
        }
      }
    }

    Array<tir::IterVar>
    splitAndReorder(te::Stage s, std::unordered_map<int, tir::IterVar> idx2iv,
                    std::unordered_map<int, Array<IntImm>> tileFactor,
                    std::vector<std::vector<int>> loopOrder,
                    Map<tir::IterVar, Bool> *isSpatial_p = NULL)
    {
      size_t n_level;
      std::vector<std::unordered_map<int, tir::IterVar>> unOrderedloopOrder;
      const te::ComputeOpNode *cop = s->op.as<te::ComputeOpNode>();
      Map<tir::IterVar, Bool> isSpatial;
      if (isSpatial_p)
      {
        for (auto iv : cop->axis)
          isSpatial.Set(iv, Bool(true));
        for (auto iv : cop->reduce_axis)
          isSpatial.Set(iv, Bool(false));
      }
      for (size_t i = 0; i < loopOrder.size(); i++)
        unOrderedloopOrder.push_back(std::unordered_map<int, tir::IterVar>());
      for (auto idxFactor : tileFactor)
      {
        int idx = idxFactor.first;
        if (!idx2iv.count(idx))
          continue;
        n_level = 0;
        Array<IntImm> factors = idxFactor.second;
        // CHECK(factors.size() == loopOrder.size() - 1) << idx2iv[idx];
        for (auto factor : factors)
        {
          if (idx2iv[idx]->dom.defined() && idx2iv[idx]->dom->extent.defined())
          {
            if (const tir::IntImmNode *t = idx2iv[idx]->dom->extent.as<tir::IntImmNode>())
            {
              if (t->value <= factor->value)
              {
                break;
              }
            }
          }
          if (factor->value == 1)
          {
            n_level += 1;
            continue;
          }
          tir::IterVar outer, inner;
          s.split(idx2iv[idx], factor, &outer, &inner);
          if (isSpatial_p)
          {
            isSpatial.Set(outer, isSpatial[idx2iv[idx]]);
            isSpatial.Set(inner, isSpatial[idx2iv[idx]]);
          }
          unOrderedloopOrder[n_level][idx] = inner;
          idx2iv[idx] = outer;
          n_level += 1;
        }
        unOrderedloopOrder[n_level][idx] = idx2iv[idx];
      }
      Array<tir::IterVar> ret;
      for (size_t level = unOrderedloopOrder.size(); level > 0; level--)
      {
        for (auto idx : loopOrder[level - 1])
        {
          if (unOrderedloopOrder[level - 1].count(idx))
            ret.push_back(unOrderedloopOrder[level - 1][idx]);
        }
      }
      if (isSpatial_p)
        *isSpatial_p = isSpatial;
      return ret;
    }

    te::Schedule ScheduleContextNode::run(int i, te::Schedule sch, te::Operation op,
                                          Array<tir::IterVar> tensorizeAxes,
                                          te::TensorIntrin intrin, String code,
                                          String path)
    {
      auto factor = data[i];

      const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
      std::unordered_map<int, tvm::tir::IterVar> idx2iv;
      int idx = 0;
      for (auto iv : cop->axis)
        idx2iv[idx++] = iv;
      for (auto iv : cop->reduce_axis)
        idx2iv[idx++] = iv;
      std::unordered_map<int, Array<IntImm>> firstOpTileFactor;
      for (auto it : factor.factor.tileSize)
      {
        int idx = it.first;
        firstOpTileFactor[idx] = Array<IntImm>();
        // firstOpTileFactor[idx].push_back(it.second[0]);
        Array<IntImm> factor = it.second;
        for (size_t i = 1; i < factor.size(); i++)
        {
          // CHECK(factor[i]->value % factor[i - 1]->value == 0);
          firstOpTileFactor[idx].push_back(IntImm(
              DataType::Int(32), (factor[i]->value + factor[i - 1]->value - 1) /
                                     factor[i - 1]->value));
        }
      }
      Map<tir::IterVar, Bool> isSpatial;
      Array<tir::IterVar> loopOrder = splitAndReorder(
          sch[op], idx2iv, firstOpTileFactor, factor.factor.loopOrder, &isSpatial);
      for (auto item : firstOpTileFactor)
      {
        std::cout << idx2iv[item.first] << ": " << item.second << std::endl;
      }
      for (auto iv : tensorizeAxes)
        loopOrder.push_back(iv);
      sch[op].reorder(loopOrder);
      Array<tir::IterVar> reduceAxes, fuseAxes;
      for (size_t i = 0; i < loopOrder.size(); i++)
      {
        if (!isSpatial[loopOrder[i]])
          reduceAxes.push_back(loopOrder[i]);
        else
        {
          for (size_t j = i; j < loopOrder.size(); j++)
          {
            if (isSpatial[loopOrder[j]])
              fuseAxes.push_back(loopOrder[j]);
            else
              break;
          }
          break;
        }
      }
      return sch;
      std::cout << "reduce size: " << reduceAxes.size()
                << ", spatial size: " << fuseAxes.size() << std::endl;
      tir::IterVar outerfuse, outerfuseouter, outerfuseinner;
      sch[op].fuse(fuseAxes, &outerfuse);
      std::cout << "parallelism: " << fuseAxes.size() << " " << parallelism
                << std::endl;
      sch[op].split_by_nparts(outerfuse, parallelism, &outerfuseouter,
                              &outerfuseinner);
      sch[op].parallel(outerfuseouter);
      Array<tir::IterVar> newOrder;
      newOrder.push_back(outerfuseouter);
      for (auto iv : reduceAxes)
        newOrder.push_back(iv);
      newOrder.push_back(outerfuseinner);
      sch[op].reorder(newOrder);
      sch[op].tensorize(tensorizeAxes[0], intrin);
      sch[op].pragma(loopOrder[0], "import_llvm", tir::StringImm(code));
      if (path.size())
      {
        std::ofstream outfile;
        outfile.open(path, std::ios_base::app); // append instead of overwrite
        outfile << i << " ";
        for (auto cost_ : factor.costs)
        {
          outfile << cost_ << " ";
        }
        // outfile << "tileSize: ";
        // for (auto item : factor.factor.tileSize) {
        //   outfile << "(" << idx2iv[item.first]->var << "," << item.second << "),
        //   ";
        // }
        // outfile << "loopOrder: ";
        // for (auto item : factor.factor.loopOrder) {
        //   outfile << "(";
        //   for (auto it : item)
        //     outfile << idx2iv[it]->var << ", ";
        //   outfile << ")";
        // }
        outfile << std::endl;
      }
      return sch;
    }

    void ScheduleSecondOpCPU(te::Schedule sch, CPUTensorizeContext ctx,
                             CPUTensorizeParam tensorize_param)
    {

      CHECK(ctx->state->second_op->num_outputs() == 1U);
      te::Operation cur_op = ctx->state->second_op;
      std::unordered_map<int, tir::IterVar> idx2iv;
      std::unordered_map<int, tir::IterVar> idx2iv_outer;
      Array<tir::IterVar> outerParallel;
      Array<tir::IterVar> allIters = ctx->GetAllIters(cur_op);
      for (size_t i = 0; i < allIters.size(); i++)
        idx2iv[i] = allIters[i];

      // 1. split the outer loops
      FusionInfo fusionInfo = tensorize_param->fusionInfo;
      for (size_t i = 0; i < fusionInfo.secondOpOuterIndices.size(); i++)
      {
        int idx = fusionInfo.secondOpOuterIndices[i];
        tir::IterVar outer, inner;
        int factor = fusionInfo.secondOpOuterTilingFactors[i];
        tir::IterVar iv = idx2iv[idx];
        if (const IntImmNode *t = iv->dom->extent.as<tir::IntImmNode>())
        {
          if (t->value <= factor)
            continue;
        }
        sch[cur_op].split(iv, factor, &outer, &inner);
        idx2iv[idx] = inner;
        idx2iv_outer[idx] = outer;
      }

      for (auto idxFactor : fusionInfo.parallelFactor)
      {
        tir::IterVar iv, outer, inner;
        if (!idx2iv_outer.count(idxFactor.first))
          continue;
        iv = idx2iv_outer.at(idxFactor.first);
        sch[cur_op].split_by_nparts(iv, idxFactor.second, &outer, &inner);
        outerParallel.push_back(outer);
        idx2iv_outer[idxFactor.first] = inner;
      }

      // 2. the body tiling
      Array<tir::IterVar> bodyLoops = splitAndReorder(
          sch[cur_op], idx2iv, tensorize_param->secondOpTilingFactor,
          tensorize_param->secondOpLoopOrder);
      Array<tir::IterVar> commonLoops = splitAndReorder(
          sch[cur_op], idx2iv_outer, tensorize_param->commonTilingFactor,
          tensorize_param->commonLoopOrder);
      Array<tir::IterVar> tensorizeLoops = ctx->GetTensorizeIters(cur_op);

      // 3. reorder
      Array<tir::IterVar> loopOrder;
      for (auto iv : outerParallel)
        loopOrder.push_back(iv);
      for (auto iv : commonLoops)
        loopOrder.push_back(iv);
      for (auto iv : bodyLoops)
        loopOrder.push_back(iv);
      for (auto iv : tensorizeLoops)
        loopOrder.push_back(iv);
      sch[cur_op].reorder(loopOrder);

      // 4. schedule the outer axis parallel
      tir::IterVar outerFused;
      sch[cur_op].fuse(outerParallel, &outerFused);
      sch[cur_op].parallel(outerFused);

      // 5. tensorize
      PackedIntrinsic pintrin = ctx->state->tensorize_intrinsics.at(cur_op);
      sch[cur_op].tensorize(tensorizeLoops[0], pintrin->compute_intrinsic);
      CHECK(ctx->state->tensorize_impl.count(cur_op));
      sch[cur_op].pragma(outerFused, "import_llvm", ctx->state->tensorize_impl.at(cur_op));

      // 6. set attach
      if (commonLoops.size())
        ctx->first_frag_attach_axis = *commonLoops.rbegin();
      else
        ctx->first_frag_attach_axis = outerFused;
      te::Operation interPathOp = ctx->InterPathRootOp();
      if (interPathOp.defined())
      {
        ctx->path_attach_axis = getAttachAxis(sch, cur_op, interPathOp, ctx->first_frag_attach_axis, bodyLoops);
        ctx->path_attach_tensor = cur_op;
      }
      ctx->first_frag_attach_op = cur_op;
      return;
    }

    bool tensorizeOp(te::Schedule sch, te::Operation op, CPUTensorizeContext ctx)
    {
      if (!ctx->state->tensorize_intrinsics.count(op) || !op.as<te::ComputeOpNode>())
      {
        return false;
      }
      auto cop = op.as<te::ComputeOpNode>();
      auto intrin = ctx->state->tensorize_intrinsics.at(op);
      auto iters = ctx->state->tensorize_iters.at(op);
      auto impl = ctx->state->tensorize_impl.at(op);
      CHECK(intrin.defined() && iters.defined() && impl.defined());
      Array<tir::IterVar> all_iters;
      for (auto iv : cop->axis)
      {
        bool isTensorize = false;
        for (auto tiv : iters)
        {
          if (iv.same_as(tiv))
          {
            isTensorize = true;
            break;
          }
        }
        if (!isTensorize)
          all_iters.push_back(iv);
      }
      for (auto iv : iters)
      {
        all_iters.push_back(iv);
      }
      sch[op].reorder(all_iters);
      sch[op].tensorize(iters[0], intrin->compute_intrinsic);
      sch[op].pragma(all_iters[0], "import_llvm", impl);
      return true;
    }

    void ScheduleInterPath(te::Schedule sch, CPUTensorizeContext ctx,
                           CPUTensorizeParam tensorize_param)
    {
      te::Operation root_op = ctx->InterPathRootOp();
      if (!root_op.defined())
        return;

      CHECK(ctx->path_attach_axis.defined());

      CHECK(ctx->path_attach_tensor.defined() && ctx->path_attach_axis.defined());
      sch[root_op].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
      tensorizeOp(sch, root_op, ctx);
      std::cout << "path_attach_axis: " << ctx->path_attach_axis << std::endl;
      for (auto op : ctx->InterPathNonRootOps())
      {
        if (tensorizeOp(sch, op, ctx))
        {
          sch[op].compute_at(sch[ctx->first_frag_attach_op], ctx->first_frag_attach_axis);
        }
        else
        {
          CHECK(ctx->CanInline(op));
          sch[op].compute_inline();
        }
      }
    }

    void ScheduleFirstOpCPU(te::Schedule sch, CPUTensorizeContext ctx,
                            CPUTensorizeParam tensorize_param)
    {
      CHECK(ctx->state->first_op->num_outputs() == 1U);
      te::Operation cur_op = ctx->state->first_op;
      std::unordered_map<int, tir::IterVar> idx2iv;
      size_t idx = 0;
      for (auto it : ctx->GetAllIters(cur_op))
      {
        idx2iv[idx++] = it;
      }

      // 1. compute_at second op
      sch[cur_op].compute_at(sch[ctx->state->second_op],
                             ctx->first_frag_attach_axis);
      // 2. the body tiling
      Array<tir::IterVar> bodyLoops =
          splitAndReorder(sch[cur_op], idx2iv, tensorize_param->firstOpTilingFactor,
                          tensorize_param->firstOpLoopOrder);
      // Array<tir::IterVar> bodyLoops;

      Array<tir::IterVar> tensorizeLoops = ctx->GetTensorizeIters(cur_op);

      // for (auto it : ctx->GetAllIters(cur_op)){
      //   bool found = false;
      //   for (auto tiv: tensorizeLoops){
      //     if (it.same_as(tiv)){
      //       found = true;
      //       break;
      //     }
      //   }
      //   if (!found) bodyLoops.push_back(it);
      // }

      // 3. reorder
      Array<tir::IterVar> loopOrder;
      for (auto iv : bodyLoops)
        loopOrder.push_back(iv);
      for (auto iv : tensorizeLoops)
        loopOrder.push_back(iv);

      sch[cur_op].reorder(loopOrder);
      // std::cout << "first Op LoopOrder: " << loopOrder << std::endl;

      // 4. tensorize
      PackedIntrinsic pintrin = ctx->state->tensorize_intrinsics.at(cur_op);
      sch[cur_op].tensorize(tensorizeLoops[0], pintrin->compute_intrinsic);
      CHECK(ctx->state->tensorize_impl.count(cur_op));
      sch[cur_op].pragma(loopOrder[0], "import_llvm", ctx->state->tensorize_impl.at(cur_op));

      return;
    }

    te::Schedule TensorizeCPU(Layer layer, TensorizeHyperFusionState state,
                              hardware::HardwareParam cpu_param,
                              CPUTensorizeParam tensorize_param)
    {
      te::Schedule sch = te::create_schedule(layer->ops);
      CPUTensorizeContext ctx = CPUTensorizeContext(layer, state);
      /*
       * Schedule epilogue
       * Currently, we treat epilogue as a separate kernel
       */
      ScheduleEpilogue(sch, ctx, tensorize_param);
      /*
       * Schedule second op tensor
       * The second tensor determines the overall parallelism
       */
      ScheduleSecondOpCPU(sch, ctx, tensorize_param);
      /*
       * Schedule second op's prologue
       */
      for (auto op_list : ctx->state->second_op_prologue)
      {
        bool isFirstOp = true;
        for (auto op : op_list)
        {
          if (isFirstOp)
          {
            SmartParallelSchedule(sch[op], tensorize_param);
          }
          else
          {
            CHECK(ctx->CanInline(op));
            sch[op].compute_inline();
          }
          isFirstOp = false;
        }
      }

      /*
       * Schedule inter path
       */
      ScheduleInterPath(sch, ctx, tensorize_param);
      /*
       * Schedule first op
       */
      ScheduleFirstOpCPU(sch, ctx, tensorize_param);
      /*
       * Schedule first op's prologue
       */
      ScheduleFirstOpPrologue(sch, ctx, tensorize_param);
      /*
       * import the intrinsic
       */

      return sch;
    }

    CostAndFactor ScheduleHelper(
        SingleCubicScheduleFactor factor, // the base factor,
        Map<tir::Var, IntImm> bounds, std::vector<double> cacheSizes,
        std::unordered_map<int, tir::Var> idx2var,
        Array<AccessFunction> accessFunctions, Map<tir::Var, IntImm> fixedTileSize,
        std::vector<double> delayedWeight_init, std::vector<double> weightPerTensor,
        std::vector<double> weightPerCacheLevel, const size_t beginCacheLevel,
        const size_t endCacheLevel, size_t bytePerEle,
        std::string searchType = "stochastic", std::string mode = "best",
        std::vector<CostAndFactor> *data = NULL, bool verbose = false, std::string prefix = "")
    {
      std::vector<CostAndFactor> candidates;
      std::vector<double> footprints;
      int cnt = 0;
      std::function<void(std::vector<double>, size_t, std::vector<double>,
                         SingleCubicScheduleFactor)>
          dfsTileSize = [&](std::vector<double> cost, size_t cacheLevel,
                            std::vector<double> delayedWeight,
                            SingleCubicScheduleFactor factors)
      {
        if (verbose)
          std::cout << "cacheLevel:" << cacheLevel << std::endl;
        if (cacheLevel >= endCacheLevel)
        {
          double cost_ = 0;
          for (size_t i = 0; i < accessFunctions.size(); i++)
            cost_ += footprints[i] * delayedWeight[i] * bytePerEle;
          CHECK(cost.size());
          auto p = cost.rbegin();
          *p += cost_;
          cnt += 1;
          candidates.push_back({cost, factors});
          return;
        }

        std::vector<std::pair<int, int>> baseTileSize;
        size_t skip;
        auto getDM = [&accessFunctions, &weightPerCacheLevel, &delayedWeight,
                      &cacheLevel, &weightPerTensor, &skip, &bounds,
                      &footprints, &verbose,
                      &bytePerEle](Map<tir::Var, IntImm> tileSize,
                                   double *delayedDM_p = NULL)
        {
          if (verbose)
          {
            std::cout << "delayed weight:" << std::endl;
            for (auto dlwt : delayedWeight)
              std::cout << dlwt << " ";
            std::cout << std::endl;
            std::cout << "skip" << skip << std::endl;
            std::cout << "cachelevel: " << cacheLevel << std::endl;
            std::cout << "tileSize: " << tileSize << std::endl;
          }
          Map<tir::Var, FloatImm> quotients;
          double prod = 1;
          for (auto varext : tileSize)
          {
            tir::Var var = varext.first;
            int ext = varext.second->value;
            double quotient = (bounds[var]->value + ext - 1) / ext;
            quotients.Set(var, FloatImm(DataType::Float(64), quotient));
            prod *= quotients[var]->value;
          }
          if (verbose)
            std::cout << "quotients: " << quotients << std::endl;
          double dm = 0;
          double delayedDM = 0;
          for (size_t i = 0; i < accessFunctions.size(); i++)
          {
            double dm_ = footprints[i];
            for (auto var : accessFunctions[i]->absentVars)
              dm_ *= quotients[var]->value;

            double weight = delayedWeight[i];

            delayedDM += dm_ * weight;

            if (i != skip)
              weight += weightPerCacheLevel[cacheLevel] * weightPerTensor[i];
            if (verbose)
            {
              std::cout << "dm" << i << " : " << dm_ << " ";
              std::cout << accessFunctions[i]->absentVars << " " << footprints[i] << std::endl;
              std::cout << "weight" << weight << std::endl;
            }
            dm += dm_ * weight;
          }
          if (delayedDM_p)
            *delayedDM_p = delayedDM * bytePerEle;
          return dm * bytePerEle;
        };
        auto getFP = [&accessFunctions,
                      &bytePerEle](Map<tir::Var, IntImm> tileSize)
        {
          double fp = 0;
          for (size_t i = 0; i < accessFunctions.size(); i++)
          {
            for (auto fp_ : accessFunctions[i]->getFootprint(tileSize))
              fp += (double)fp_ * bytePerEle;
          }
          return fp;
        };
        struct cacheLevelFactor
        {
          double cost;
          Map<tir::Var, IntImm> TileSize;
          double occupancy;
          double delayedCost;
          double curCost;
          double fp;
          cacheLevelFactor(double cost_, Map<tir::Var, IntImm> TileSize_,
                           double occupancy_, double delayedCost_,
                           double curCost_, double fp_)
              : cost(cost_), TileSize(TileSize_), occupancy(occupancy_),
                delayedCost(delayedCost_), curCost(curCost_), fp(fp_) {}
        };
        std::vector<cacheLevelFactor> best;
        size_t beam;
        std::function<void(int, Map<tir::Var, IntImm>)>
            dfsTileSizeOfCacheLevel = [&](size_t idx,
                                          Map<tir::Var, IntImm> tileSize)
        {
          if (verbose)
            std::cout << "idx: " << idx << std::endl;
          if (mode == "survey" && best.size() >= beam)
            return;
          int pos = baseTileSize[idx].first;
          if (idx == baseTileSize.size())
          {
            // do the evaluation
            if (verbose)
              std::cout << "tileSize" << tileSize << std::endl;
            double fp = getFP(tileSize);
            if (fp > cacheSizes[cacheLevel])
              return;
            double delayedDM;
            double dm = getDM(tileSize, &delayedDM);
            double occupancy = getFP(tileSize) / cacheSizes[cacheLevel];
            cacheLevelFactor newItem = {dm, tileSize, occupancy, delayedDM,
                                        dm - delayedDM, fp};
            if (mode == "best")
            {
              if (verbose)
                std::cout << "best.size(): " << best.size() << std::endl;
              size_t rank = 0;
              while (best.size() > rank && best[rank].cost < dm)
                rank++;
              best.insert(best.begin() + rank, newItem);
              while (best.size() > beam)
                best.pop_back();
            }
            else if (mode == "survey")
            {
              best.push_back(newItem);
            }
            return;
          }
          for (size_t i = idx; i < baseTileSize.size(); i++)
          {
            tileSize.Set(idx2var[baseTileSize[i].first],
                         IntImm(DataType::Int(32), baseTileSize[i].second));
          }
          if (getFP(tileSize) > cacheSizes[cacheLevel])
            return;

          if (mode == "best")
          {
            for (size_t i = idx; i < baseTileSize.size(); i++)
            {
              int bound = bounds[idx2var[baseTileSize[i].first]]->value;
              tileSize.Set(idx2var[baseTileSize[i].first],
                           IntImm(DataType::Int(32), bound));
            }
            if (best.size() == beam &&
                getDM(tileSize) >= best[beam - 1].cost)
              return;
          }
          int bound =
              (bounds[idx2var[pos]]->value + baseTileSize[idx].second - 1) /
              baseTileSize[idx].second;
          size_t n_trial =
              (searchType == "stochastic" ? std::min(bound, 10) : bound);
          for (size_t trial = 0; trial < n_trial; trial++)
          {
            int ext = (trial + 1) * baseTileSize[idx].second;
            if (searchType == "stochastic")
              ext = (random() % bound + 1) * baseTileSize[idx].second;
            tileSize.Set(idx2var[pos], IntImm(DataType::Int(32), ext));
            dfsTileSizeOfCacheLevel(idx + 1, tileSize);
          }
          return;
        };
        Map<tir::Var, IntImm> tileSizeInit;
        for (auto varExt : fixedTileSize)
          tileSizeInit.Set(varExt.first, varExt.second);
        for (auto idx : factors.tileSize)
        {
          baseTileSize.push_back(
              {idx.first, idx.second[idx.second.size() - 1]->value});
        }
        if (verbose)
        {
          std::cout << "baseTileSize:\n";
          for (auto idxext : baseTileSize)
          {
            std::cout << "(" << idx2var[idxext.first] << ", " << idxext.second
                      << "), ";
            std::cout << std::endl;
          }
        }
        beam = 20;
        for (size_t i = 0; i < accessFunctions.size(); i++)
        {
          // determine the delayed weight
          skip = i;
          dfsTileSizeOfCacheLevel(0, tileSizeInit);

          std::vector<double> newDelayedWeight;
          for (size_t j = 0; j < accessFunctions.size(); j++)
          {
            if (j != i)
              newDelayedWeight.push_back(0);
            else
              newDelayedWeight.push_back(weightPerTensor[i] *
                                         weightPerCacheLevel[cacheLevel]);
          }

          Array<tir::Var> absentVars = accessFunctions[i]->absentVars;
          auto isAbsentVar = [&absentVars](tir::Var var_)
          {
            for (auto var : absentVars)
              if (var.same_as(var_))
                return true;
            return false;
          };
          std::vector<int> newLoopOrder;
          std::vector<int> innerOrder;
          for (auto it : baseTileSize)
          {
            int idx = it.first;
            if (isAbsentVar(idx2var[idx]))
              innerOrder.push_back(idx);
            else
              newLoopOrder.push_back(idx);
          }
          newLoopOrder.insert(newLoopOrder.end(), innerOrder.begin(),
                              innerOrder.end());

          for (auto item : best)
          {
            auto tileSize = item.TileSize;
            SingleCubicScheduleFactor newFactors = factors;
            for (auto &kv : newFactors.tileSize)
            {
              kv.second.push_back(tileSize[idx2var[kv.first]]);
            }
            newFactors.skip.push_back(i);
            newFactors.loopOrder.push_back(newLoopOrder);
            newFactors.log[prefix + "fp" + std::to_string(cacheLevel)] = item.fp;
            newFactors.log[prefix + "cost" + std::to_string(cacheLevel)] = item.curCost;
            auto newCost = cost;
            CHECK(newCost.size());
            auto p = newCost.rbegin();
            *p += item.delayedCost;
            newCost.push_back(item.curCost);
            newFactors.cacheOccupancy.push_back(item.occupancy);
            dfsTileSize(newCost, cacheLevel + 1, newDelayedWeight, newFactors);
          }
        }
      };
      // check factor.tileSize doesn't have fixed vars
      for (auto iv : factor.tileSize)
      {
        CHECK(idx2var.count(iv.first)) << "idx2var not contain " << iv.first;
        tir::Var var = idx2var[iv.first];
        CHECK(fixedTileSize.count(var) == 0)
            << "fixed tilesize exist in init factor";
      }

      for (auto acf : accessFunctions)
      {
        int fp = 0;
        for (auto fp_ : acf->getFootprint(bounds))
          fp += fp_;
        footprints.push_back(fp);
      }
      dfsTileSize({0.0}, beginCacheLevel, delayedWeight_init, factor);
      std::sort(candidates.begin(), candidates.end(),
                [](CostAndFactor &a, CostAndFactor &b)
                { return a.sum < b.sum; });

      if (!candidates.size())
      {
        if (verbose)
        {
          std::cout << std::endl;
          std::cout << "bounds: " << std::endl;
          std::cout << bounds << std::endl;
          std::cout << "fixed tileSize" << std::endl;
          std::cout << fixedTileSize << std::endl;
          std::cout << "begin/end: " << beginCacheLevel << ":" << endCacheLevel << std::endl;
          std::cout << "cacheSizes: " << cacheSizes.size() << std::endl;
          std::cout << "weightPerCacheLevel.size()" << weightPerCacheLevel.size() << std::endl;
          std::cout << "delayedWeightInit.size()" << delayedWeight_init.size() << std::endl;
          std::cout << "weightPerTensor.size()" << weightPerTensor.size() << std::endl;
          std::cout
              << "acf.size()" << accessFunctions.size() << std::endl;
          std::cout << "access functions:";
          for (auto acf : accessFunctions)
          std::cout << acf << " " << acf->absentVars << std::endl;
          for (auto acf : accessFunctions)
            std::cout << "access_indices" << acf->access_indices
                      << "absentVars: " << acf->absentVars
                      << ", presentVars: " << acf->presentVars << std::endl;
          std::cout << std::endl;
          std::cout << "tensorWeight:";
          for (auto tswt : weightPerTensor)
            std::cout << tswt << " ";
          std::cout << std::endl;
          std::cout << "weightPerCacheLevel:";
          for (auto wpcl : weightPerCacheLevel)
            std::cout << wpcl << " ";
          std::cout << std::endl;
          for (auto idxFactors : factor.tileSize)
          {
            int idx = idxFactors.first;
            auto factors = idxFactors.second;
            CHECK(bounds.count(idx2var[idx]));
            CHECK(factors.size()) << idx2var[idx] << bounds[idx2var[idx]] << " has no factor";
            std::cout << "[" << idx2var[idx] << " , (" << factors[0] << ", " << bounds[idx2var[idx]] << ")], ";
          }
          std::cout << std::endl;
        }
        LOG(WARNING) << "no valid candidates found";
        return CostAndFactor(false);
      }
      if (data)
      {
        std::vector<CostAndFactor> candidates_;
        if (mode == "best")
          candidates_.push_back(candidates[0]);
        else if (mode == "survey")
          for (size_t i = 0; i < candidates.size();
               i += std::max(1, (int)candidates.size() / 10))
          {
            candidates_.push_back(candidates[i]);
          }
        *data = candidates_;
      }
      return candidates[0];
    }

    CostAndFactor scheduleSingleCubic(OpHyperState op, Map<tir::Var, IntImm> bounds,
                                      hardware::HardwareParam hw_param,
                                      const size_t fusionCacheLevel,
                                      size_t bytePerEle,
                                      std::string searchType = "stochastic",
                                      std::string mode = "best",
                                      std::vector<CostAndFactor> *data = NULL, std::string prefix = "")
    {
      std::vector<double> weightPerCacheLevel, weightPerTensor, delayedWeight_init;
      std::vector<CostAndFactor> candidates;
      std::unordered_map<int, tir::Var> idx2var;
      Array<AccessFunction> accessFunctions;
      Map<tir::Var, IntImm> fixedTileSize;

      idx2var = op->getIdx2var();
      for (auto bdwidth : hw_param->cacheBandwidth)
        weightPerCacheLevel.push_back(1 / (double)bdwidth);
      for (auto acf : op->ReadAccessFunctions())
      {
        accessFunctions.push_back(acf);
        weightPerTensor.push_back(hw_param->tensorWeight[0]);
        delayedWeight_init.push_back(0);
      }
      accessFunctions.push_back(op->WriteAccessFunctions());
      weightPerTensor.push_back(hw_param->tensorWeight[1]);
      delayedWeight_init.push_back(
          0); // currently we do not have register level reuse

      // hard-code the shape configuration for tensorize
      SingleCubicScheduleFactor factor;
      std::unordered_map<int, Array<IntImm>> &tileFactorInit = factor.tileSize;
      std::vector<int> L1LoopOrder;
      std::vector<int> innerOrder;
      for (auto iv : op->getAllIters())
      {
        if (iv->iv_type != IV_Type::TENSORIZE)
        {
          tileFactorInit[iv->index] = Array<IntImm>();
          tileFactorInit[iv->index].push_back(IntImm(DataType::Int(32), 1));
          if (iv->iv_type != IV_Type::REDUCE)
            L1LoopOrder.push_back(iv->index);
          else
            innerOrder.push_back(iv->index);
        }
        else
          fixedTileSize.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      }
      for (auto idx : innerOrder)
        L1LoopOrder.push_back(idx);
      factor.loopOrder.push_back(L1LoopOrder);
      factor.skip.push_back(accessFunctions.size() - 1);
      return ScheduleHelper(factor, bounds, hw_param->cacheSizePerThread, idx2var,
                            accessFunctions, fixedTileSize, delayedWeight_init,
                            weightPerTensor, weightPerCacheLevel, 1,
                            fusionCacheLevel, bytePerEle, searchType, mode, data, false, prefix);
    }

    CostAndFactor scheduleCommonLoop(OpHyperState op1, OpHyperState op2,
                                     hardware::HardwareParam hw_param,
                                     FusionInfo fusionInfo, const size_t bytePerEle,
                                     const std::string searchType,
                                     const std::string mode,
                                     std::vector<CostAndFactor> *data, std::string prefix = "")
    {
      SingleCubicScheduleFactor factor;
      std::unordered_map<int, tir::Var> idx2var = op2->getIdx2var();
      std::vector<double> weightPerTensor, weightPerCacheLevel, delayedWeight;
      Map<tir::Var, IntImm> innerBounds;

      Array<AccessFunction> accessFunctions;

      // init the innerbounds
      for (auto iv : fusionInfo.lower_bounds_for_upper_cache_level)
      {
        innerBounds.Set(iv.first, iv.second);
      }
      // the init tileSize
      std::unordered_map<int, Array<IntImm>> &tileFactorSize = factor.tileSize;
      std::vector<int> loopOrderInit;
      for (auto idx : fusionInfo.secondOpOuterIndices)
      {
        tileFactorSize[idx] = Array<IntImm>();
        tir::Var var = idx2var[idx];
        tileFactorSize[idx].push_back(innerBounds[var]);
        innerBounds.erase(var);
        loopOrderInit.push_back(idx);
      }
      factor.loopOrder.push_back(loopOrderInit);

      // init of weightPerCacheLevel
      for (auto bdwidth : hw_param->cacheBandwidth)
        weightPerCacheLevel.push_back(1 / bdwidth);

      auto sharedPairs = share_axis_analysis(op1->op, op2->op);
      Map<tir::Var, tir::Var> varMap;
      for (auto arr : sharedPairs)
      {
        CHECK(arr.size() == 2);
        varMap.Set(arr[0]->var, arr[1]->var);
      }
      // init of access function
      Array<tir::Var> op2AllVars = op2->getAllVars();
      for (auto acf : op1->ReadAccessFunctions())
      {
        acf->repalceVars(varMap);
        Array<tir::Var> newAbsentVar;
        for (auto var : op2AllVars)
        {
          bool isAbsent = true;
          for (auto var_ : acf->presentVars)
          {
            if (var.same_as(var_))
            {
              isAbsent = false;
              break;
            }
          }
          if (isAbsent)
            newAbsentVar.push_back(var);
        }
        acf->setAbsentVars(newAbsentVar);
        accessFunctions.push_back(acf);
        weightPerTensor.push_back(hw_param->tensorWeight[0]);
      }
      accessFunctions.push_back(op2->WriteAccessFunctions());
      weightPerTensor.push_back(hw_param->tensorWeight[1]);
      // judge whether the access function is shared
      for (auto acf : op2->ReadAccessFunctions())
      {
        auto inAbsentVar = [&acf](tir::Var var)
        {
          for (auto var_ : acf->absentVars)
            if (var_.same_as(var))
              return true;
          return false;
        };
        bool isShared = true;
        for (auto iv : op2->getAllIters())
        {
          if (iv->iv_type == IV_Type::SECONDSPATIAL)
          {
            if (!inAbsentVar(iv->originVar))
            {
              isShared = false;
              break;
            }
          }
        }
        if (!isShared)
        {
          accessFunctions.push_back(acf);
          weightPerTensor.push_back(hw_param->tensorWeight[0]);
        }
      }
      for (auto acf : accessFunctions)
        delayedWeight.push_back(0);

      return ScheduleHelper(
          factor, fusionInfo.boundsAfterParallel, hw_param->cacheSizePerThread,
          idx2var, accessFunctions, innerBounds, delayedWeight, weightPerTensor,
          weightPerCacheLevel, fusionInfo.fusionLevel + 1,
          hw_param->cacheSizePerThread.size(), bytePerEle, searchType, mode, data, false, prefix);
    }
    CPUTensorizeParam buildCPUTensorizeParam(SerialFusionState sfs,
                                             hardware::HardwareParam hw_param,
                                             int bytePerEle,
                                             FusionInfo fusionInfo,
                                             std::string mode,
                                             std::string search_type)
    {
      OpHyperState op1, op2;
      std::tie(op1, op2) = sfs->getCubicOpPair();
      CostAndFactor op1caf =
          scheduleSingleCubic(op1, fusionInfo.upper_bounds_for_lower_cache_level, hw_param, fusionInfo.fusionLevel,
                              bytePerEle, search_type, mode, NULL, "op1");
      CostAndFactor op2caf =
          scheduleSingleCubic(op2, fusionInfo.upper_bounds_for_lower_cache_level, hw_param, fusionInfo.fusionLevel,
                              bytePerEle, search_type, mode, NULL, "op2");
      CostAndFactor commCaf = scheduleCommonLoop(
          op1, op2, hw_param, fusionInfo, bytePerEle, search_type, mode, NULL, "comm");
      printf("[buildCPUTensorizeParam]: %d, %d, %d\n", op1caf.valid, op2caf.valid, commCaf.valid);
      if (!op1caf.valid || !op2caf.valid || !commCaf.valid)
        return CPUTensorizeParam(false);

      SingleCubicScheduleFactor op1Schedule = op1caf.factor;
      SingleCubicScheduleFactor op2Schedule = op2caf.factor;
      SingleCubicScheduleFactor commSchedule = commCaf.factor;

      auto add_fusion_level_loops = [](
        const std::vector<std::pair<int, int>>&ivs,
        SingleCubicScheduleFactor&sch){
        if (!ivs.size()) return;
        sch.loopOrder.push_back({});
        for (auto kv: ivs){
          sch.loopOrder.back().push_back(kv.first);
          sch.tileSize[kv.first].push_back(IntImm(DataType::Int(32), kv.second));
        }
      };

      add_fusion_level_loops(fusionInfo.op1InnerIvs, op1Schedule);
      add_fusion_level_loops(fusionInfo.op2InnerIvs, op2Schedule);

      auto tileSize2factor =
          [](const SingleCubicScheduleFactor &sch,
             std::unordered_map<int, Array<IntImm>> &tileFactor)
      {
        for (auto it : sch.tileSize)
        {
          int idx = it.first;
          Array<IntImm> factor = it.second;
          tileFactor[idx] = Array<IntImm>();
          for (size_t i = 1; i < factor.size(); i++)
          {
            CHECK(factor[i]->value % factor[i - 1]->value == 0);
            tileFactor[idx].push_back(IntImm(
                DataType::Int(32), factor[i]->value / factor[i - 1]->value));
          }
        }
      };
      std::unordered_map<int, Array<IntImm>> firstOpTileFactor, secondOpTileFactor,
          commonLoopTileFcator;
      tileSize2factor(op1Schedule, firstOpTileFactor);
      tileSize2factor(op2Schedule, secondOpTileFactor);
      tileSize2factor(commSchedule, commonLoopTileFcator);

      // construct the occupancy
      Array<FloatImm> occupancy;
      for (auto occupancy_ : op1caf.factor.cacheOccupancy)
        occupancy.push_back(FloatImm(DataType::Float(32), occupancy_));
      for (auto occupancy_ : op1caf.factor.cacheOccupancy)
        occupancy.push_back(FloatImm(DataType::Float(32), occupancy_));
      occupancy.push_back(FloatImm(DataType::Float(32), fusionInfo.cacheOccupancy));
      for (auto occupancy_ : commSchedule.cacheOccupancy)
        occupancy.push_back(FloatImm(DataType::Float(32), occupancy_));

      std::unordered_map<std::string, double> log;
      for (auto kv : commCaf.factor.log)
      {
        log[kv.first] = kv.second;
      }
      for (auto kv : op1caf.factor.log)
      {
        log[kv.first] = kv.second;
      }
      for (auto kv : op2caf.factor.log)
      {
        log[kv.first] = kv.second;
      }
      for (auto &cost : op1caf.costs)
        cost = cost * fusionInfo.outerCost;
      for (auto &cost : op2caf.costs)
        cost = cost * fusionInfo.outerCost;
      for (auto &cost : commCaf.costs)
        cost *= fusionInfo.maxThreadIter;

      return CPUTensorizeParam(
          op1, op2, hw_param->num_groups, op1Schedule.loopOrder,
          op2Schedule.loopOrder, commSchedule.loopOrder, firstOpTileFactor,
          secondOpTileFactor, commonLoopTileFcator, fusionInfo, op1caf.costs,
          op2caf.costs, commCaf.costs, occupancy, log);
    }

    ScheduleContext SingleOpSchedule(te::Operation op,
                                     Array<tir::IterVar> tensorizeAxes,
                                     hardware::HardwareParam hw_param,
                                     String searchType = "stochastic",
                                     String mode = "best")
    {
      OpHyperState op_ = buildOpHyperState(op, 0);
      OpHyperStateNode::tensorizeAxes = tensorizeAxes;
      Map<tir::Var, IntImm> bounds;
      for (auto iv : op_->getAllIters())
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      std::vector<CostAndFactor> candidates;
      scheduleSingleCubic(op_, bounds, hw_param, 3, 4, searchType, mode,
                          &candidates);
      return ScheduleContext(candidates, hw_param->num_groups);
    }

    te::Schedule FusionContextNode::run(int i, te::Schedule sch, bool verbose)
    {
      CHECK(0 <= i && i < (int)schParams.size())
          << "run " << i << " out of range "
          << "[0, " << schParams.size() << ")";
      CPUTensorizeParam schParam = schParams[i];
      FusionInfo info = schParam->fusionInfo;
      if (path.size())
      {
        std::ofstream outfile;
        outfile.open(path, std::ios_base::app); // append instead of overwrite
        outfile << i << " ";
        double sum = 0;
        for (auto cost_ : schParam->costs)
        {
          sum += cost_;
          outfile << cost_ << " ";
        }
        outfile << sum;
        outfile << std::endl;
      }
      std::cout << "run " << i << " fusionLevel:" << schParam->fusionInfo.fusionLevel << std::endl;
      if (verbose)
      {
        double cost_sum = 0;
        for (auto cost : schParam->costs)
        {
          cost_sum += cost;
          std::cout << cost << " ";
        }
        std::cout << cost_sum << std::endl;
      }
      OpHyperState op1, op2;
      std::tie(op1, op2) = sfs->getCubicOpPair();
      std::unordered_map<int, tir::Var> op2_idx2var = op2->getIdx2var();
      if (verbose)
        std::cout << "outer indices: ";
      for (size_t i = 0; i < info.secondOpOuterIndices.size(); i++)
      {
        tir::Var var = op2_idx2var[info.secondOpOuterIndices[i]];
        int factor = info.secondOpOuterTilingFactors[i];
        if (verbose)
          std::cout << "(" << var << ": " << factor << "), ";
      }
      if (verbose)
      {
        std::cout << std::endl;
        std::cout << schParam << std::endl;
        std::cout << std::setw(12) << "cacheSize";
        for (size_t i = 1; i < hw_param->cacheSizes.size(); i++)
          std::cout << std::setw(12) << hw_param->cacheSizes[i];
        std::cout << std::endl;
        std::cout << std::setw(12) << "bandwidth";
        for (size_t i = 1; i < hw_param->cacheBandwidth.size(); i++)
          std::cout << std::setw(12) << hw_param->cacheBandwidth[i];
        std::cout << std::endl;
        std::cout << "fusionInfo\n";
        std::cout << schParam->fusionInfo;
        std::cout << std::endl;
      }
      return TensorizeCPU(layer, state, hw_param, schParam);
    }

    TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<CPUTensorizeParamNode>([](const ObjectRef &node,
                                                ReprPrinter *p)
                                             {
          auto *op = static_cast<const CPUTensorizeParamNode *>(node.get());
          p->PrintIndent();
          p->stream << "---------------CPU Tensorize Param------------------\n";
          std::unordered_map<int, tir::Var> firstOpIdx2var = op->op1->getIdx2var();
          std::unordered_map<int, tir::Var> secondOpIdx2var = op->op2->getIdx2var();
          p->stream << "fusion level: " << op->fusionInfo.fusionLevel << std::endl;
          // auto bounds = op->fusionInfo.bounds;
          auto parallel_bounds = op->fusionInfo.boundsAfterParallel;
          p->stream << "tileSize: " << std::endl;
          std::cout << std::setw(12) << "" << std::setw(12) << "L1" << std::setw(12) << "L2" << std::setw(12) << "L3" << std::setw(12) << "B" << std::setw(12) << "P" << std::endl;
          for (auto& idx_factor : op->firstOpTilingFactor)
          {
            auto var = firstOpIdx2var[idx_factor.first];
            p->stream << std::setw(12) << "op1." + std::string(var->name_hint);
            int tileSize = 1;
            int level = 0;
            for (auto factor : idx_factor.second)
            {
              tir::Var innerMostVar = firstOpIdx2var[*op->firstOpLoopOrder[level + 1].rbegin()];
              tileSize *= factor->value;
              if (var.same_as(innerMostVar))
                p->stream << std::setw(12) << std::to_string(tileSize) + "*";
              else
                p->stream << std::setw(12) << tileSize;
              level++;
            }
            tileSize = op->fusionInfo.lower_bounds_for_upper_cache_level.at(var)->value;
            p->stream << std::setw(12) << tileSize;
            p->stream << std::endl;
          }
          for (auto& idx_factor : op->secondOpTilingFactor)
          {
            auto var = secondOpIdx2var[idx_factor.first];
            p->stream << std::setw(12) << "op2." + std::string(var->name_hint);
            int tileSize = 1;
            int level = 0;
            for (auto factor : idx_factor.second)
            {
              tir::Var innerMostVar = secondOpIdx2var[*op->secondOpLoopOrder[level + 1].rbegin()];
              tileSize *= factor->value;
              if (var.same_as(innerMostVar))
                p->stream << std::setw(12) << std::to_string(tileSize) + "*";
              else
                p->stream << std::setw(12) << tileSize;
              level++;
            }
            tileSize = op->fusionInfo.lower_bounds_for_upper_cache_level.at(var)->value;
            p->stream << std::setw(12) << tileSize;
            if (op->commonTilingFactor.count(idx_factor.first)) {
              auto commFactor = op->commonTilingFactor.at(idx_factor.first);
              level = 0;
              for (auto factor : commFactor)
              {
                tir::Var innerMostVar = secondOpIdx2var[*op->commonLoopOrder[level + 1].rbegin()];
                tileSize *= factor->value;
                if (var.same_as(innerMostVar))
                  p->stream << std::setw(12) << std::to_string(tileSize) + "*";
                else
                  p->stream << std::setw(12) << tileSize;
                level++;
              }
            }
            p->stream << std::setw(12) << op->fusionInfo.boundsAfterParallel.at(var);
            if (op->fusionInfo.parallelFactor.count(idx_factor.first))
            {
              p->stream << std::setw(12) << "P" + std::to_string(op->fusionInfo.parallelFactor.at(idx_factor.first)->value);
            }
            p->stream << std::endl;
          }
          p->stream << std::setw(12) << "cost:";
          for (size_t i = 1; i < op->firstOpCosts.size(); i++)
          {
            double cost = op->firstOpCosts[i] + op->secondOpCosts[i];
            p->stream << std::setw(12) << cost;
          }
          p->stream << std::setw(12) << op->fusionInfo.cost;
          for (size_t i = 1; i < op->commCosts.size(); i++)
            p->stream << std::setw(12) << op->commCosts[i];
          p->stream << std::endl;
          p->stream << std::setw(12) << "fp:";
          for (size_t i = 1; i < op->firstOpCosts.size(); i++)
          {
            std::string key = "op1fp" + std::to_string(i);
            CHECK(op->log.count(key));
            p->stream << std::setw(12) << op->log.at(key);
          }
          p->stream << std::setw(12) << op->fusionInfo.memUse;
          for (size_t i = 1; i < op->commCosts.size(); i++)
          {
            std::string key = "commfp" + std::to_string(i + op->fusionInfo.fusionLevel);
            CHECK(op->log.count(key));
            p->stream << std::setw(12) << op->log.at(key);
          }
          p->stream << std::endl;
          p->stream << std::setw(12) << "op2fp";
          for (size_t i = 1; i < op->secondOpCosts.size(); i++)
          {
            std::string key = "op2fp" + std::to_string(i);
            CHECK(op->log.count(key));
            p->stream << std::setw(12) << op->log.at(key);
          }
          p->stream << std::endl;
          p->stream << "\n----------------------------------------------------\n"; });

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CPUTensorizeContext")
        .set_body_typed([](Layer layer, TensorizeHyperFusionState state)
                        { return CPUTensorizeContext(layer, state); });

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildCPUTensorizeParam")
        .set_body_typed([](SerialFusionState sfs, FusionChoice fusionChoice,
                           hardware::HardwareParam hw_param, int bytePerEle)
                        {
          FusionInfo fusionInfo;
          fusionInfo.fusionLevel = fusionChoice->fusionResult->fusionLevel;
          IterGraph ig = buildIterGraph(sfs);
          ig->setConfig(hw_param, bytePerEle);
          ig->setFusionLevel(fusionInfo.fusionLevel);
          ig->setFusion(fusionChoice->fusionItem);
          double occupancy, parallelism, memUse, cost;
          bool valid;
          std::tie(valid, cost) = ig->getCost(&occupancy, &parallelism, &memUse);
          fusionInfo.cacheOccupancy = occupancy;
          fusionInfo.cost = cost;
          fusionInfo.n_block = ig->getNumOfBlocks();
          fusionInfo.outerCost = ig->_outerCost;
          fusionInfo.maxThreadIter = ig->_maxThreadIter;
          fusionInfo.parallelism = parallelism;
          for (auto i : fusionChoice->secondOpOuterIndices)
            fusionInfo.secondOpOuterIndices.push_back(i->value);
          for (auto i : fusionChoice->secondOpOuterTilingFactors)
            fusionInfo.secondOpOuterTilingFactors.push_back(i->value);
          fusionInfo.lower_bounds_for_upper_cache_level =
            fusionInfo.upper_bounds_for_lower_cache_level = ig->bounds;
          fusionInfo.valid = true;
          fusionInfo.boundsAfterParallel = ig->_boundsAfterParallel;
          fusionInfo.parallelFactor = ig->_parallelSchedule;
          fusionInfo.memUse = memUse;
          CHECK(OpHyperStateNode::tensorizeAxes.defined());
          fusionInfo.computation =
              ig->getFirstOpWorkload() + ig->getSecondOpWorkload();
          return buildCPUTensorizeParam(sfs, hw_param, bytePerEle, fusionInfo); });

    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeCPU")
        .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                           hardware::HardwareParam cpu_param,
                           CPUTensorizeParam tensorize_param)
                        { return TensorizeCPU(layer, state, cpu_param, tensorize_param); });
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.SingleOpSchedule")
        .set_body_typed([](te::Operation op, Array<tir::IterVar> tensorizeAxes,
                           hardware::HardwareParam hw_param, String searchType,
                           String mode)
                        { return SingleOpSchedule(op, tensorizeAxes, hw_param, searchType, mode); });
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.run")
        .set_body_method<ScheduleContext>(&ScheduleContextNode::run);
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.runFusion")
        .set_body_method<FusionContext>(&FusionContextNode::run);
  } // namespace auto_tensorize

} // namespace ditto