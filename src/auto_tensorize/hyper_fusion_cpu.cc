#include <auto_compute/graph.h>
#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/searchDriver.h>
#include <auto_tensorize/hyper_fusion.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/state.h>
#include <stack>
#include <tvm/driver/driver_api.h>
#include <tvm/te/schedule_pass.h>
namespace ditto {

namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(TensorizeHyperFusionStateNode);
TVM_REGISTER_NODE_TYPE(CPUTensorizeContextNode);
TVM_REGISTER_NODE_TYPE(CPUTensorizeParamNode);
TVM_REGISTER_NODE_TYPE(FusionContextNode);

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

tir::IterVar CPUTensorizeContextNode::FuseAll(te::Schedule sch,
                                              te::Operation op) {
  te::Operation sop = sch[op]->op;
  const te::ComputeOpNode *cop = sop.as<te::ComputeOpNode>();
  Array<tir::IterVar> axis = cop->axis;
  tir::IterVar fused;
  sch[op].fuse(axis, &fused);
  return fused;
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

bool CPUTensorizeContextNode::isBatchLikeDim(const te::Operation &op,
                                             const tir::IterVar iv) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  CHECK(cop->body.size() == 1) << "Only expect one body.\n";
  IsBatchLikeDim checker;
  std::vector<int> ret;
  return checker.is_batch(cop->body[0], iv->var);
}

bool CPUTensorizeContextNode::isSpatial(te::Operation op, tir::IterVar iv) {
  const te::ComputeOpNode *cur_op = op.as<te::ComputeOpNode>();
  CHECK(cur_op);
  for (auto iv_ : cur_op->axis)
    if (iv.same_as(iv_))
      return true;
  return false;
}

CPUTensorizeContext::CPUTensorizeContext(Layer layer,
                                         TensorizeHyperFusionState state) {
  auto node = make_object<CPUTensorizeContextNode>();
  node->layer = layer;
  node->state = state;
  data_ = node;
}

/*!
 * \brief Get the tensorize iters
 */
Array<tir::IterVar>
CPUTensorizeContextNode::GetTensorizeIters(const te::Operation &op) {
  CHECK(this->state->tensorize_iters.count(op));
  Array<tir::IterVar> iters = this->state->tensorize_iters.at(op);
  return iters;
}

/*!
 * \brief Get the tensorize iters
 */
Array<te::IterVar>
CPUTensorizeContextNode::GetAllIters(const te::Operation &op) {
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
CPUTensorizeContextNode::GetSecondOpOuterIndexAndSplitFactor() {
  std::vector<int> indices;
  for (auto i : state->secondOpOuterIndices) {
    indices.push_back(i->value);
  }
  return {indices, state->secondOpOuterTileFactors};
}
/*!
 * \brief if the axis of op is spatial
 */
std::pair<Array<tir::IterVar>, Array<tir::IterVar>>
CPUTensorizeContextNode::splitSpatialWithReduce(const te::Operation op,
                                                Array<tir::IterVar> ivs) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop);
  Array<tir::IterVar> Spatial, Reduce;
  for (auto iv : ivs) {
    bool isSpatial = false;
    for (auto iv_s : cop->axis)
      if (iv == iv_s) {
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
    OpHyperState op1, OpHyperState op2, int parallelism,
    std::vector<std::vector<int>> firstOpLoopOrder,
    std::vector<std::vector<int>> secondOpLoopOrder,
    std::vector<std::vector<int>> commonLoopOrder,
    std::unordered_map<int, Array<IntImm>> firstOpTilingFactor,
    std::unordered_map<int, Array<IntImm>> secondOpTilingFactor,
    std::unordered_map<int, Array<IntImm>> commonTilingFactor,
    FusionInfo fusionInfo, std::vector<double> firstOpCosts,
    std::vector<double> secondOpCosts, std::vector<double> commCosts,
    Array<FloatImm> cacheOccupancy) {
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
  data_ = node;
}

ScheduleContext::ScheduleContext(std::vector<CostAndFactor> data,
                                 int parallelism) {
  auto node = make_object<ScheduleContextNode>();
  node->data = data;
  node->size = data.size();
  node->parallelism = parallelism;
  data_ = node;
}
TVM_REGISTER_NODE_TYPE(ScheduleContextNode);

void ScheduleEpilogue(te::Schedule sch, CPUTensorizeContext ctx,
                      CPUTensorizeParam tensorize_param) {
  te::Operation cur_op;
  if (ctx->HasEpilogue()) {
    cur_op = ctx->EpilogueRootOp();
    auto iv = ctx->FuseAll(sch, cur_op);
    tir::IterVar outer, inner;
    sch[cur_op].split_by_nparts(iv, tensorize_param->parallelism, &outer,
                                &inner);
    sch[cur_op].parallel(outer);
    Array<te::Operation> remain_ops = ctx->EpilogueNonRootOps();
    // the remaining non-root ops should be inlined
    for (auto op : remain_ops) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
    }
  }
}

Array<tir::IterVar>
splitAndReorder(te::Stage s, std::unordered_map<int, tir::IterVar> idx2iv,
                std::unordered_map<int, Array<IntImm>> tileFactor,
                std::vector<std::vector<int>> loopOrder,
                Map<tir::IterVar, Bool> *isSpatial_p = NULL) {
  size_t n_level;
  std::vector<std::unordered_map<int, tir::IterVar>> unOrderedloopOrder;
  const te::ComputeOpNode *cop = s->op.as<te::ComputeOpNode>();
  Map<tir::IterVar, Bool> isSpatial;
  if (isSpatial_p) {
    for (auto iv : cop->axis)
      isSpatial.Set(iv, Bool(true));
    for (auto iv : cop->reduce_axis)
      isSpatial.Set(iv, Bool(false));
  }
  for (size_t i = 0; i < loopOrder.size(); i++)
    unOrderedloopOrder.push_back(std::unordered_map<int, tir::IterVar>());

  for (auto idxFactor : tileFactor) {
    int idx = idxFactor.first;
    n_level = 0;
    Array<IntImm> factors = idxFactor.second;
    CHECK(factors.size() == loopOrder.size() - 1) << idx2iv[idx];
    for (auto factor : factors) {
      tir::IterVar outer, inner;
      s.split(idx2iv[idx], factor, &outer, &inner);
      if (isSpatial_p) {
        isSpatial.Set(outer, isSpatial[idx2iv[idx]]);
        isSpatial.Set(inner, isSpatial[idx2iv[idx]]);
      }
      unOrderedloopOrder[n_level][idx] = inner;
      idx2iv[idx] = outer;
      n_level += 1;
    }
    CHECK(n_level == loopOrder.size() - 1) << "nlevel is " << n_level;
    unOrderedloopOrder[n_level][idx] = idx2iv[idx];
  }
  Array<tir::IterVar> ret;
  for (size_t level = unOrderedloopOrder.size(); level > 0; level--) {
    for (auto idx : loopOrder[level - 1]) {
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
                                      String path) {
  auto factor = data[i];
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  std::unordered_map<int, tvm::tir::IterVar> idx2iv;
  int idx = 0;
  for (auto iv : cop->axis)
    idx2iv[idx++] = iv;
  for (auto iv : cop->reduce_axis)
    idx2iv[idx++] = iv;
  std::unordered_map<int, Array<IntImm>> firstOpTileFactor;
  for (auto it : factor.factor.tileSize) {
    int idx = it.first;
    firstOpTileFactor[idx] = Array<IntImm>();
    // firstOpTileFactor[idx].push_back(it.second[0]);
    Array<IntImm> factor = it.second;
    for (size_t i = 1; i < factor.size(); i++) {
      // CHECK(factor[i]->value % factor[i - 1]->value == 0);
      firstOpTileFactor[idx].push_back(IntImm(
          DataType::Int(32), (factor[i]->value + factor[i - 1]->value - 1) /
                                 factor[i - 1]->value));
    }
  }
  Map<tir::IterVar, Bool> isSpatial;
  Array<tir::IterVar> loopOrder = splitAndReorder(
      sch[op], idx2iv, firstOpTileFactor, factor.factor.loopOrder, &isSpatial);
  for (auto iv : tensorizeAxes)
    loopOrder.push_back(iv);
  sch[op].reorder(loopOrder);
  Array<tir::IterVar> reduceAxes, fuseAxes;
  for (size_t i = 0; i < loopOrder.size(); i++) {
    if (!isSpatial[loopOrder[i]])
      reduceAxes.push_back(loopOrder[i]);
    else {
      for (size_t j = i; j < loopOrder.size(); j++) {
        if (isSpatial[loopOrder[j]])
          fuseAxes.push_back(loopOrder[j]);
        else
          break;
      }
      break;
    }
  }
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
  if (path.size()) {
    std::ofstream outfile;
    outfile.open(path, std::ios_base::app); // append instead of overwrite
    outfile << i << " ";
    for (auto cost_ : factor.costs) {
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
                         CPUTensorizeParam tensorize_param) {

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
  for (size_t i = 0; i < fusionInfo.secondOpOuterIndices.size(); i++) {
    int idx = fusionInfo.secondOpOuterIndices[i];
    tir::IterVar outer, inner;
    int factor = fusionInfo.secondOpOuterTilingFactors[i];
    tir::IterVar iv = idx2iv[idx];
    sch[cur_op].split(iv, factor, &outer, &inner);
    idx2iv[idx] = inner;
    idx2iv_outer[idx] = outer;
  }

  for (auto idxFactor : fusionInfo.parallelFactor) {
    tir::IterVar iv, outer, inner;
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

  // 6. set attach
  ctx->path_attach_axis = commonLoops[commonLoops.size() - 1];
  ctx->first_frag_attach_axis = ctx->path_attach_axis;
  ctx->path_attach_tensor = cur_op;
  ctx->secondOpOuterMostAxis = outerFused;
  return;
}

void ScheduleInterPath(te::Schedule sch, CPUTensorizeContext ctx,
                       CPUTensorizeParam tensorize_param) {
  te::Operation root_op = ctx->InterPathRootOp();
  // sch[root_op].set_scope("shared");
  CHECK(ctx->path_attach_tensor.defined() && ctx->path_attach_axis.defined());
  sch[root_op].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
  for (auto op : ctx->InterPathNonRootOps()) {
    CHECK(ctx->CanInline(op));
    sch[op].compute_inline();
  }
}

void ScheduleFirstOpCPU(te::Schedule sch, CPUTensorizeContext ctx,
                        CPUTensorizeParam tensorize_param) {
  CHECK(ctx->state->first_op->num_outputs() == 1U);
  te::Operation cur_op = ctx->state->first_op;
  std::unordered_map<int, tir::IterVar> idx2iv;
  size_t idx = 0;
  for (auto it : ctx->GetAllIters(cur_op)) {
    idx2iv[idx++] = it;
  }

  // 1. compute_at second op
  sch[cur_op].compute_at(sch[ctx->state->second_op],
                         ctx->first_frag_attach_axis);
  // 2. the body tiling
  Array<tir::IterVar> bodyLoops =
      splitAndReorder(sch[cur_op], idx2iv, tensorize_param->firstOpTilingFactor,
                      tensorize_param->firstOpLoopOrder);

  Array<tir::IterVar> tensorizeLoops = ctx->GetTensorizeIters(cur_op);

  // 3. reorder
  Array<tir::IterVar> loopOrder;
  for (auto iv : bodyLoops)
    loopOrder.push_back(iv);
  for (auto iv : tensorizeLoops)
    loopOrder.push_back(iv);

  sch[cur_op].reorder(loopOrder);

  // 4. tensorize
  PackedIntrinsic pintrin = ctx->state->tensorize_intrinsics.at(cur_op);
  sch[cur_op].tensorize(tensorizeLoops[0], pintrin->compute_intrinsic);

  // 6. set attach
  ctx->first_prologue_shared_attach_axis = bodyLoops[bodyLoops.size() - 1];
  ctx->firstOpOuterMostAxis = loopOrder[0];
  return;
}

te::Schedule TensorizeCPU(Layer layer, TensorizeHyperFusionState state,
                          hardware::HardwareParam cpu_param,
                          CPUTensorizeParam tensorize_param,
                          tir::StringImm code) {
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
  for (auto op_list : ctx->state->second_op_prologue) {
    bool isFirstOp = false;
    for (auto op : op_list) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
      if (!isFirstOp)
        sch[op].compute_at(sch[ctx->path_attach_tensor], ctx->path_attach_axis);
      else
        sch[op].compute_inline();
      isFirstOp = true;
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
  for (auto op_list : ctx->state->first_op_prologue) {
    for (auto op : op_list) {
      CHECK(ctx->CanInline(op));
      sch[op].compute_inline();
    }
  }
  /*
   * import the intrinsic
   */
  sch[state->second_op].pragma(ctx->secondOpOuterMostAxis, "import_llvm", code);
  sch[state->first_op].pragma(ctx->firstOpOuterMostAxis, "import_llvm", code);
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
    std::vector<CostAndFactor> *data = NULL, bool verbose = false) {
  // std::cout << "begin schedule helper with param: \n";
  // for (auto idxFactors: factor.tileSize){
  //   int idx = idxFactors.first;
  //   auto factors = idxFactors.second;
  //   std::cout << "[" << idx2var[idx] << " , (" << factors[0] << ", " <<
  //   bounds[idx2var[idx]] << ")], ";
  // }
  // std::cout << std::endl;
  // std::cout << "bounds: " << std::endl;
  // std::cout << bounds << std::endl;
  // std::cout << "fixed tileSize" << std::endl;
  // std::cout << fixedTileSize << std::endl;
  // std::cout << "begin/end: " << beginCacheLevel << ":" << endCacheLevel <<
  // std::endl; std::cout << "cacheSizes: " << cacheSizes.size() << std::endl;
  // std::cout << "weightPerCacheLevel.size()" << weightPerCacheLevel.size() <<
  // std::endl; std::cout << "delayedWeightInit.size()" <<
  // delayedWeight_init.size() << std::endl; std::cout <<
  // "weightPerTensor.size()" << weightPerTensor.size() << std::endl; std::cout
  // << "acf.size()" << accessFunctions.size() << std::endl;
  if (verbose) {
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
  }

  std::vector<CostAndFactor> candidates;
  std::vector<double> footprints;
  int cnt = 0;
  std::function<void(std::vector<double>, size_t, std::vector<double>,
                     SingleCubicScheduleFactor)>
      dfsTileSize = [&](std::vector<double> cost, size_t cacheLevel,
                        std::vector<double> delayedWeight,
                        SingleCubicScheduleFactor factors) {
        if (verbose)
          std::cout << "cacheLevel:" << cacheLevel << std::endl;
        if (cacheLevel >= endCacheLevel) {
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
                                   double *delayedDM_p = NULL) {
          if (verbose) {
            std::cout << "delayed weight:" << std::endl;
            for (auto dlwt : delayedWeight)
              std::cout << dlwt << " ";
            std::cout << std::endl;
            std::cout << "skip" << skip << std::endl;
            std::cout << "cachelevel: " << cacheLevel << std::endl;
            std::cout << "bounds" << std::endl;
            std::cout << "tileSize: " << tileSize << std::endl;
          }
          Map<tir::Var, FloatImm> quotients;
          double prod = 1;
          for (auto varext : tileSize) {
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
          for (size_t i = 0; i < accessFunctions.size(); i++) {
            double dm_ = footprints[i];
            for (auto var : accessFunctions[i]->absentVars)
              dm_ *= quotients[var]->value;

            double weight = delayedWeight[i];

            delayedDM += dm_ * weight;

            if (i != skip)
              weight += weightPerCacheLevel[cacheLevel] * weightPerTensor[i];
            if (verbose)
              std::cout << "dm" << i << " : " << dm_ << std::endl;
            dm += dm_ * weight;
          }
          if (delayedDM_p)
            *delayedDM_p = delayedDM * bytePerEle;
          return dm * bytePerEle;
        };
        auto getFP = [&accessFunctions,
                      &bytePerEle](Map<tir::Var, IntImm> tileSize) {
          int fp = 0;
          for (size_t i = 0; i < accessFunctions.size(); i++) {
            for (auto fp_ : accessFunctions[i]->getFootprint(tileSize))
              fp += fp_ * bytePerEle;
          }
          return fp;
        };
        struct cacheLevelFactor {
          double cost;
          Map<tir::Var, IntImm> TileSize;
          double occupancy;
          double delayedCost;
          double curCost;
          cacheLevelFactor(double cost_, Map<tir::Var, IntImm> TileSize_,
                           double occupancy_, double delayedCost_,
                           double curCost_)
              : cost(cost_), TileSize(TileSize_), occupancy(occupancy_),
                delayedCost(delayedCost_), curCost(curCost_) {}
        };
        std::vector<cacheLevelFactor> best;
        size_t beam;
        std::function<void(int, Map<tir::Var, IntImm>)>
            dfsTileSizeOfCacheLevel = [&](size_t idx,
                                          Map<tir::Var, IntImm> tileSize) {
              if (verbose)
                std::cout << "idx: " << idx << std::endl;
              if (mode == "survey" && best.size() >= beam)
                return;
              int pos = baseTileSize[idx].first;
              if (idx == baseTileSize.size()) {
                // do the evaluation
                if (verbose)
                  std::cout << "tileSize" << tileSize << std::endl;
                int fp = getFP(tileSize);
                if (fp > cacheSizes[cacheLevel])
                  return;
                double delayedDM;
                double dm = getDM(tileSize, &delayedDM);
                double occupancy = getFP(tileSize) / cacheSizes[cacheLevel];
                cacheLevelFactor newItem = {dm, tileSize, occupancy, delayedDM,
                                            dm - delayedDM};
                if (mode == "best") {
                  if (verbose)
                    std::cout << "best.size(): " << best.size() << std::endl;
                  size_t rank = 0;
                  while (best.size() > rank && best[rank].cost < dm)
                    rank++;
                  best.insert(best.begin() + rank, newItem);
                  while (best.size() > beam)
                    best.pop_back();
                } else if (mode == "survey") {
                  best.push_back(newItem);
                }
                return;
              }
              for (size_t i = idx; i < baseTileSize.size(); i++) {
                tileSize.Set(idx2var[baseTileSize[i].first],
                             IntImm(DataType::Int(32), baseTileSize[i].second));
              }
              if (getFP(tileSize) > cacheSizes[cacheLevel])
                return;

              if (mode == "best") {
                for (size_t i = idx; i < baseTileSize.size(); i++) {
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
              for (size_t trial = 0; trial < n_trial; trial++) {
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
        for (auto idx : factors.tileSize) {
          baseTileSize.push_back(
              {idx.first, idx.second[idx.second.size() - 1]->value});
        }
        if (verbose) {
          std::cout << "baseTileSize:\n";
          for (auto idxext : baseTileSize) {
            std::cout << "(" << idx2var[idxext.first] << ", " << idxext.second
                      << "), ";
            std::cout << std::endl;
          }
        }
        if (cacheLevel == 1)
          beam = 100;
        else
          beam = 5;
        for (size_t i = 0; i < accessFunctions.size(); i++) {
          // determine the delayed weight
          skip = i;
          dfsTileSizeOfCacheLevel(0, tileSizeInit);

          // std::cout << "cacheLevel: " << cacheLevel << ", bestCost"
          //           << best[0].cost << std::endl;
          // std::cout << "the ratio: "
          //           << getFP(best[0].TileSize) /
          //                  (double)hw_param->cacheSizes[cacheLevel]
          //           << std::endl;
          // CHECK(best.size() && best[0].cost < 1e9)
          //     << "best cost is " << best[0].cost;

          std::vector<double> newDelayedWeight;
          for (size_t j = 0; j < accessFunctions.size(); j++) {
            if (j != i)
              newDelayedWeight.push_back(0);
            else
              newDelayedWeight.push_back(weightPerTensor[i] *
                                         weightPerCacheLevel[cacheLevel]);
          }

          Array<tir::Var> absentVars = accessFunctions[i]->absentVars;
          auto isAbsentVar = [&absentVars](tir::Var var_) {
            for (auto var : absentVars)
              if (var.same_as(var_))
                return true;
            return false;
          };
          std::vector<int> newLoopOrder;
          std::vector<int> innerOrder;
          for (auto it : baseTileSize) {
            int idx = it.first;
            if (isAbsentVar(idx2var[idx]))
              innerOrder.push_back(idx);
            else
              newLoopOrder.push_back(idx);
          }
          newLoopOrder.insert(newLoopOrder.end(), innerOrder.begin(),
                              innerOrder.end());

          for (auto item : best) {
            auto tileSize = item.TileSize;
            SingleCubicScheduleFactor newFactors = factors;
            for (auto &kv : newFactors.tileSize) {
              kv.second.push_back(tileSize[idx2var[kv.first]]);
            }
            newFactors.skip.push_back(i);
            newFactors.loopOrder.push_back(newLoopOrder);
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
  for (auto iv : factor.tileSize) {
    CHECK(idx2var.count(iv.first)) << "idx2var not contain " << iv.first;
    tir::Var var = idx2var[iv.first];
    CHECK(fixedTileSize.count(var) == 0)
        << "fixed tilesize exist in init factor";
  }

  for (auto acf : accessFunctions) {
    int fp = 0;
    for (auto fp_ : acf->getFootprint(bounds))
      fp += fp_;
    footprints.push_back(fp);
  }
  std::cout << std::endl;
  dfsTileSize({0.0}, beginCacheLevel, delayedWeight_init, factor);
  std::sort(candidates.begin(), candidates.end(),
            [](CostAndFactor &a, CostAndFactor &b) { return a.sum < b.sum; });
  if (data) {
    std::vector<CostAndFactor> candidates_;
    if (mode == "best")
      candidates_.push_back(candidates[0]);
    else if (mode == "survey")
      for (size_t i = 0; i < candidates.size();
           i += std::max(1, (int)candidates.size() / 10)) {
        candidates_.push_back(candidates[i]);
      }
    *data = candidates_;
  }
  CHECK(candidates.size()) << "no valid candidates" << std::endl;
  return candidates[0];
}

CostAndFactor scheduleSingleCubic(OpHyperState op, Map<tir::Var, IntImm> bounds,
                                  hardware::HardwareParam hw_param,
                                  const size_t fusionCacheLevel,
                                  size_t bytePerEle,
                                  std::string searchType = "stochastic",
                                  std::string mode = "best",
                                  std::vector<CostAndFactor> *data = NULL) {
  std::vector<double> weightPerCacheLevel, weightPerTensor, delayedWeight_init;
  std::vector<CostAndFactor> candidates;
  std::unordered_map<int, tir::Var> idx2var;
  Array<AccessFunction> accessFunctions;
  Map<tir::Var, IntImm> fixedTileSize;

  idx2var = op->getIdx2var();
  for (auto bdwidth : hw_param->cacheBandwidth)
    weightPerCacheLevel.push_back(1 / (double)bdwidth);
  for (auto acf : op->ReadAccessFunctions()) {
    accessFunctions.push_back(acf);
    weightPerTensor.push_back(hw_param->tensorWeight[0]);
    delayedWeight_init.push_back(0);
  }
  accessFunctions.push_back(op->WriteAccessFunctions());
  weightPerTensor.push_back(hw_param->tensorWeight[1]);
  delayedWeight_init.push_back(
      0); // currently we do not have register level reuse
  // init footprint

  SingleCubicScheduleFactor factor;
  std::unordered_map<int, Array<IntImm>> &tileFactorInit = factor.tileSize;
  std::vector<int> L1LoopOrder;
  std::vector<int> innerOrder;
  for (auto iv : op->getAllIters()) {
    if (iv->iv_type != IV_Type::TENSORIZE) {
      tileFactorInit[iv->index] = Array<IntImm>();
      tileFactorInit[iv->index].push_back(IntImm(DataType::Int(32), 1));
      if (iv->iv_type != IV_Type::REDUCE)
        L1LoopOrder.push_back(iv->index);
      else
        innerOrder.push_back(iv->index);
    } else
      fixedTileSize.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
  }
  for (auto idx : innerOrder)
    L1LoopOrder.push_back(idx);
  factor.loopOrder.push_back(L1LoopOrder);
  factor.skip.push_back(accessFunctions.size() - 1);
  return ScheduleHelper(factor, bounds, hw_param->cacheSizePerThread, idx2var,
                        accessFunctions, fixedTileSize, delayedWeight_init,
                        weightPerTensor, weightPerCacheLevel, 1,
                        fusionCacheLevel, bytePerEle, searchType, mode, data);
}

CostAndFactor scheduleCommonLoop(OpHyperState op1, OpHyperState op2,
                                 hardware::HardwareParam hw_param,
                                 FusionInfo fusionInfo, const size_t bytePerEle,
                                 const std::string searchType,
                                 const std::string mode,
                                 std::vector<CostAndFactor> *data) {
  SingleCubicScheduleFactor factor;
  std::unordered_map<int, tir::Var> idx2var = op2->getIdx2var();
  std::vector<double> weightPerTensor, weightPerCacheLevel, delayedWeight;
  Map<tir::Var, IntImm> innerBounds;

  Array<AccessFunction> accessFunctions;

  // init the innerbounds
  for (auto iv : fusionInfo.bounds) {
    innerBounds.Set(iv.first, iv.second);
  }
  // the init tileSize
  std::unordered_map<int, Array<IntImm>> &tileFactorSize = factor.tileSize;
  std::vector<int> loopOrderInit;
  for (auto idx : fusionInfo.secondOpOuterIndices) {
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
  for (auto arr : sharedPairs) {
    CHECK(arr.size() == 2);
    varMap.Set(arr[0]->var, arr[1]->var);
  }
  // init of access function
  Array<tir::Var> op2AllVars = op2->getAllVars();
  for (auto acf : op1->ReadAccessFunctions()) {
    acf->repalceVars(varMap);
    Array<tir::Var> newAbsentVar;
    for (auto var : op2AllVars) {
      bool isAbsent = true;
      for (auto var_ : acf->presentVars) {
        if (var.same_as(var_)) {
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
  for (auto acf : op2->ReadAccessFunctions()) {
    auto inAbsentVar = [&acf](tir::Var var) {
      for (auto var_ : acf->absentVars)
        if (var_.same_as(var))
          return true;
      return false;
    };
    bool isShared = true;
    for (auto iv : op2->getAllIters()) {
      if (iv->iv_type == IV_Type::SECONDSPATIAL) {
        if (!inAbsentVar(iv->originVar)) {
          isShared = false;
          break;
        }
      }
    }
    if (!isShared) {
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
      hw_param->cacheSizePerThread.size(), bytePerEle, searchType, mode, data);
}
CPUTensorizeParam buildCPUTensorizeParam(SerialFusionState sfs,
                                         hardware::HardwareParam hw_param,
                                         int bytePerEle,
                                         FusionInfo fusionInfo) {

  OpHyperState op1, op2;
  std::tie(op1, op2) = sfs->getCubicOpPair();
  Map<tir::Var, IntImm> bounds = fusionInfo.bounds;
  CostAndFactor op1caf =
      scheduleSingleCubic(op1, bounds, hw_param, fusionInfo.fusionLevel,
                          bytePerEle, "normal", "best", NULL);
  CostAndFactor op2caf =
      scheduleSingleCubic(op2, bounds, hw_param, fusionInfo.fusionLevel,
                          bytePerEle, "normal", "best", NULL);
  CostAndFactor commCaf = scheduleCommonLoop(
      op1, op2, hw_param, fusionInfo, bytePerEle, "normal", "best", NULL);

  SingleCubicScheduleFactor op1Schedule = op1caf.factor;
  SingleCubicScheduleFactor op2Schedule = op2caf.factor;
  SingleCubicScheduleFactor commSchedule = commCaf.factor;

  auto tileSize2factor =
      [](const SingleCubicScheduleFactor &sch,
         std::unordered_map<int, Array<IntImm>> &tileFactor) {
        for (auto it : sch.tileSize) {
          int idx = it.first;
          Array<IntImm> factor = it.second;
          tileFactor[idx] = Array<IntImm>();
          for (size_t i = 1; i < factor.size(); i++) {
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

  for (auto &cost : op1caf.costs)
    cost = cost * (double)fusionInfo.n_block / (double)fusionInfo.parallelism;
  for (auto &cost : op2caf.costs)
    cost = cost * (double)fusionInfo.n_block / (double)fusionInfo.parallelism;
  return CPUTensorizeParam(
      op1, op2, hw_param->num_groups, op1Schedule.loopOrder,
      op2Schedule.loopOrder, commSchedule.loopOrder, firstOpTileFactor,
      secondOpTileFactor, commonLoopTileFcator, fusionInfo, op1caf.costs,
      op2caf.costs, commCaf.costs, occupancy);
}

ScheduleContext SingleOpSchedule(te::Operation op,
                                 Array<tir::IterVar> tensorizeAxes,
                                 hardware::HardwareParam hw_param,
                                 String searchType = "stochastic",
                                 String mode = "best") {
  OpHyperState op_ = buildOpHyperState(op, 0, tensorizeAxes);
  Map<tir::Var, IntImm> bounds;
  for (auto iv : op_->getAllIters())
    bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
  std::vector<CostAndFactor> candidates;
  scheduleSingleCubic(op_, bounds, hw_param, 3, 4, searchType, mode,
                      &candidates);
  return ScheduleContext(candidates, hw_param->num_groups);
}

te::Schedule FusionContextNode::run(int i, te::Schedule sch, bool verbose) {
  CHECK(0 <= i && i < (int)schParams.size())
      << "run " << i << " out of range "
      << "[0, " << schParams.size() << ")";
  CPUTensorizeParam schParam = schParams[i];
  FusionInfo info = schParam->fusionInfo;
  if (path.size()) {
    std::ofstream outfile;
    outfile.open(path, std::ios_base::app); // append instead of overwrite
    outfile << i << " ";
    double sum = 0;
    for (auto cost_ : schParam->costs) {
      sum += cost_;
      outfile << cost_ << " ";
    }
    outfile << sum;
    outfile << std::endl;
  }
  std::cout << "run " << i << std::endl;
  if (verbose) {
    double cost_sum = 0;
    for (auto cost : schParam->costs) {
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
  for (size_t i = 0; i < info.secondOpOuterIndices.size(); i++) {
    tir::Var var = op2_idx2var[info.secondOpOuterIndices[i]];
    int factor = info.secondOpOuterTilingFactors[i];
    if (verbose)
      std::cout << "(" << var << ": " << factor << "), ";
  }
  if (verbose) {
    std::cout << std::endl;
    std::cout << schParam << std::endl;
  }
  return TensorizeCPU(layer, state, hw_param, schParam, code);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CPUTensorizeParamNode>([](const ObjectRef &node,
                                            ReprPrinter *p) {
      auto *op = static_cast<const CPUTensorizeParamNode *>(node.get());
      p->PrintIndent();
      p->stream << "---------------CPU Tensorize Param------------------\n";
      std::unordered_map<int, tir::Var> firstOpIdx2var = op->op1->getIdx2var();
      p->stream << "fusion level: " << op->fusionInfo.fusionLevel;
      p->stream << "first op tileSize: " << std::endl;
      for (auto idx_factor : op->firstOpTilingFactor)
        p->stream << "\t" << firstOpIdx2var[idx_factor.first] << ": "
                  << idx_factor.second << std::endl;
      p->stream << "first op order:" << std::endl;
      int level = 0;
      for (auto order : op->firstOpLoopOrder) {
        p->stream << "\t" << level++ << ": ";
        for (auto i : order)
          p->stream << firstOpIdx2var[i] << ", ";
        p->stream << std::endl;
      }

      std::unordered_map<int, tir::Var> secondOpIdx2var = op->op2->getIdx2var();
      p->stream << "second op tileSize: " << std::endl;
      for (auto idx_factor : op->secondOpTilingFactor)
        p->stream << "\t" << secondOpIdx2var[idx_factor.first] << ": "
                  << idx_factor.second << std::endl;
      p->stream << "second op order:" << std::endl;
      level = 0;
      for (auto order : op->secondOpLoopOrder) {
        p->stream << "\t";
        p->stream << "\t" << level++ << ": ";
        for (auto i : order)
          p->stream << secondOpIdx2var[i] << ", ";
        p->stream << std::endl;
      }

      p->stream << "common loop: " << std::endl;
      for (auto idx_factor : op->commonTilingFactor)
        p->stream << "\t" << secondOpIdx2var[idx_factor.first] << ": "
                  << idx_factor.second << std::endl;
      p->stream << "common loop order:" << std::endl;
      level = 0;
      for (auto order : op->commonLoopOrder) {
        p->stream << "\t";
        p->stream << "\t" << level++ << ": ";
        for (auto i : order)
          p->stream << secondOpIdx2var[i] << ", ";
        p->stream << std::endl;
      }
      p->stream << "firstOpCosts: ";
      for (auto cost : op->firstOpCosts)
        std::cout << cost << " ";
      std::cout << std::endl;
      p->stream << "secondOpCost: ";
      for (auto cost : op->secondOpCosts)
        std::cout << cost << " ";
      std::cout << std::endl;
      p->stream << "commonLoopCost: ";
      for (auto cost : op->commCosts)
        std::cout << cost << " ";
      std::cout << std::endl;
      p->stream << "fusionCost: " << op->fusionInfo.cost << std::endl;
      p->stream << "\n----------------------------------------------------\n";
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.CPUTensorizeContext")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state) {
      return CPUTensorizeContext(layer, state);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildCPUTensorizeParam")
    .set_body_typed([](SerialFusionState sfs, FusionChoice fusionChoice,
                       hardware::HardwareParam hw_param, int bytePerEle) {
      /*
              fusionInfo.cacheOccupancy = occupancy;
              fusionInfo.cost = dm / hw_param->cacheBandwidth[fusionLevel];
              fusionInfo.n_block = ig->getNumOfBlocks();
              fusionInfo.parallelism = std::min(hw_param->num_groups,
         (int)ig->getParallelism()); fusionInfo.secondOpOuterIndices =
         secondOpOuterIndices; fusionInfo.fusionLevel = fusionLevel;
              fusionInfo.computation = ig->getFirstOpWorkload() +
         ig->getSecondOpWorkload(); fusionInfo.bounds = ig->bounds;
              fusionInfo.valid = true;
              ig->scheduleParallel();
              fusionInfo.boundsAfterParallel = ig->_boundsAfterParallel;
              fusionInfo.parallelFactor = ig->_parallelSchedule;
              for (auto idx : secondOpUnsetIndices)
                fusionInfo.secondOpOuterTilingFactors.push_back(
                    secondOpTilingFactors[idx]);
      */
      FusionInfo fusionInfo;
      fusionInfo.fusionLevel = fusionChoice->fusionResult->fusionLevel;
      fusionInfo.cost = fusionChoice->fusionResult->dataMovement /
                        hw_param->cacheBandwidth.at(fusionInfo.fusionLevel);
      for (auto i : fusionChoice->secondOpOuterIndices)
        fusionInfo.secondOpOuterIndices.push_back(i->value);
      for (auto i : fusionChoice->secondOpOuterTilingFactors)
        fusionInfo.secondOpOuterTilingFactors.push_back(i->value);
      fusionInfo.n_block = fusionChoice->fusionResult->n_block;
      fusionInfo.parallelism = fusionChoice->fusionResult->parallelism;
      fusionInfo.valid = true;
      fusionInfo.bounds = fusionChoice->fusionResult->bounds;
      IterGraph ig = buildIterGraph(sfs, sfs->tensorizeAxes, "");
      ig->setParallel(hw_param->num_groups);
      ig->setFusionLevel(fusionInfo.fusionLevel);
      ig->setFusion(fusionChoice->fusionItem);
      ig->scheduleParallel();
      fusionInfo.boundsAfterParallel = ig->_boundsAfterParallel;
      fusionInfo.parallelFactor = ig->_parallelSchedule;
      fusionInfo.cacheOccupancy = fusionChoice->fusionResult->occupancy;
      fusionInfo.computation =
          ig->getFirstOpWorkload() + ig->getSecondOpWorkload();
      return buildCPUTensorizeParam(sfs, hw_param, bytePerEle, fusionInfo);
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.TensorizeCPU")
    .set_body_typed([](Layer layer, TensorizeHyperFusionState state,
                       hardware::HardwareParam cpu_param,
                       CPUTensorizeParam tensorize_param, String code) {
      tir::StringImm code_ = code;
      return TensorizeCPU(layer, state, cpu_param, tensorize_param, code_);
    });
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.SingleOpSchedule")
    .set_body_typed([](te::Operation op, Array<tir::IterVar> tensorizeAxes,
                       hardware::HardwareParam hw_param, String searchType,
                       String mode) {
      return SingleOpSchedule(op, tensorizeAxes, hw_param, searchType, mode);
    });
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.run")
    .set_body_method<ScheduleContext>(&ScheduleContextNode::run);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.runFusion")
    .set_body_method<FusionContext>(&FusionContextNode::run);
} // namespace auto_tensorize

} // namespace ditto