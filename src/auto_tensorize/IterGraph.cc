#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/evaluator.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/pattern.h>
#include <auto_tensorize/state.h>
#include <fstream>
#include <sys/stat.h>
#include <utils/iter_domain.h>
namespace ditto {

namespace auto_tensorize {

Share::Share(IterVar upper, IterVar lower) {
  auto n = make_object<ShareNode>();
  n->upper = upper;
  n->lower = lower;
  data_ = std::move(n);
}
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShareNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const ShareNode *>(node.get());
      p->PrintIndent();
      p->stream << "Share(";
      p->stream << "upper: ";
      p->Print(op->upper);
      p->stream << ", ";
      p->stream << "lower: ";
      p->Print(op->lower);
      p->stream << ") ";
    });
Split::Split(IterVar parent, IterVar outer, IterVar inner) {
  auto n = make_object<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  data_ = std::move(n);
}
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SplitNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const SplitNode *>(node.get());
      p->PrintIndent();
      p->stream << "Split(";
      p->stream << "outer: ";
      p->Print(op->outer);
      p->stream << ", ";
      p->stream << "inner: ";
      p->Print(op->inner);
      p->stream << ") ";
    });
IterVar::IterVar(int idx_, FACTOR ext_, IV_Type iv_type_, tvm::tir::Var name_,
                 tvm::tir::Var originVar) {
  ObjectPtr<IterVarNode> n = make_object<IterVarNode>();
  n->index = idx_;
  n->ext = ext_;
  n->iv_type = iv_type_;
  n->name = name_;
  n->originVar = originVar;
  n->shared = false;
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterVarNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const IterVarNode *>(node.get());
      p->PrintIndent();
      // p->stream << "IterVar(";
      // p->stream << "ext: " << op->ext << ", ";
      // p->stream << "type: " << (op->iv_type == IV_Type::SPATIAL? "S":"R") <<
      // ", ";
      p->stream << op->name;
    });

AccessFunction::AccessFunction(te::Operation op,
                               Array<Array<PrimExpr>> access_indices) {
  auto node = make_object<AccessFunctionNode>();
  node->op = op;
  node->access_indices = access_indices;
  data_ = node;
}
// TODO: the footprint of non-cartesian product
std::vector<int>
AccessFunctionNode::getFootprint(Map<tir::Var, IntImm> bounds) {
  int idx = 0;
  Map<tir::Var, Range> ori_ranges;
  for (auto kv : bounds)
    ori_ranges.Set(kv.first, Range(0, kv.second));
  std::vector<int> ret;
  for (auto access_index : access_indices) {
    Map<tir::Var, PrimExpr> vars_to_infer;
    for (auto index : access_index) {
      vars_to_infer.Set(tvm::tir::Var("v" + std::to_string(idx)), index);
    }
    Map<tir::Var, Range> var_range_map =
        utils::InferRange(vars_to_infer, ori_ranges);
    int fp = 1;
    for (auto kv : var_range_map)
      fp *= kv.second->extent.as<IntImmNode>()->value;
    ret.push_back(fp);
  }
  return ret;
}

int AccessFunctionNode::getWorkload(Map<tir::Var, IntImm> bounds) {
  auto cop = op.as<te::ComputeOpNode>();
  int ret = 1;
  for (auto iv : cop->axis) {
    tir::Var var = iv->var;
    ret *= bounds[var]->value;
  }
  for (auto iv : cop->reduce_axis) {
    tir::Var var = iv->var;
    ret *= bounds[var]->value;
  }
  return ret;
}
TVM_REGISTER_NODE_TYPE(IterGraphNode);
IterGraph::IterGraph(Array<IterVar> firstOpIters, Array<IterVar> secondOpIters,
                     Array<Share> sharedIterPairs,
                     Array<AccessFunction> firstOpReadAccessFuncs,
                     Array<AccessFunction> secondOpReadAccessFuncs,
                     AccessFunction firstOpWriteAccessFunc,
                     AccessFunction secondOpWriteAccessFunc,
                     int readProducerPos, te::Operation op1, te::Operation op2,
                     String path) {
  auto n = make_object<IterGraphNode>();
  n->_firstOpIters = firstOpIters;
  n->_secondOpIters = secondOpIters;
  n->_firstOpReadAccessFuncs = firstOpReadAccessFuncs;
  n->_secondOpReadAccessFuncs = secondOpReadAccessFuncs;
  n->_firstOpWriteAccessFunc = firstOpWriteAccessFunc;
  n->_secondOpWriteAccessFunc = secondOpWriteAccessFunc;

  n->firstOpIters = firstOpIters;
  n->secondOpIters = secondOpIters;
  n->firstOpNumLoops = firstOpIters.size();
  n->secondOpNumLoops = secondOpIters.size();
  n->firstOpReadAccessFuncs = firstOpReadAccessFuncs;
  n->secondOpReadAccessFuncs = secondOpReadAccessFuncs;
  n->shareRelations = sharedIterPairs;
  n->firstOpWriteAccessFunc = firstOpWriteAccessFunc;
  n->secondOpWriteAccessFunc = secondOpWriteAccessFunc;
  n->readProducerPos = readProducerPos;
  n->op1 = op1;
  n->op2 = op2;
  n->resultPath = path;
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0) {
    LOG(WARNING) << path << "exists. Results will not be written to any file.";
    n->resultPath = "";
  }
  data_ = std::move(n);
}

void IterGraphNode::setFirstOpTiling(Array<IntImm> factors_) {
  std::vector<FACTOR> factors;
  for (auto i : factors_)
    factors.push_back(i->value);
  CHECK(factors.size() == _firstOpIters.size())
      << "the length of input factors does not match first Ops private iters";
  auto findPos = [&](IterVar iv) {
    int idx = 0;
    for (auto i : _firstOpIters) {
      if (i == iv)
        return idx;
      idx++;
    }
    return -1;
  };
  bool splited = firstOpIters.size() > _firstOpIters.size();
  if (splited) {
    for (auto i : splitRelations) {
      int pos = findPos(i->parent);
      if (pos >= 0)
        i->inner->ext = factors[pos];
    }
    return;
  }
  Array<IterVar> outer, inner;
  for (size_t i = 0; i < factors.size(); i++) {
    IterVar parent = firstOpIters[i];
    CHECK(factors[i] <= parent->ext) << "tiling factor larger than axis extent";
    IterVar outer_, inner_;
    outer_ = IterVar(parent->index, -1, parent->iv_type,
                     parent->name.copy_with_suffix(".outer"), tir::Var());
    // the inner iter success the Var
    inner_ =
        IterVar(parent->index, factors[findPos(parent)], parent->iv_type,
                parent->name.copy_with_suffix(".inner"), parent->originVar);
    outer.push_back(outer_);
    inner.push_back(inner_);
    splitRelations.push_back(Split(parent, outer_, inner_));
    for (size_t j = 0; j < shareRelations.size(); j++)
      if (shareRelations[j]->upper == parent) {
        shareRelations[j]->upper = outer_;
      }
  }
  outer.insert(outer.end(), inner.begin(), inner.end());
  firstOpIters = outer;
}

void IterGraphNode::setSecondOpTiling(Array<IntImm> factors_) {
  std::vector<FACTOR> factors;
  for (auto i : factors_)
    factors.push_back(i->value);
  CHECK(factors.size() == _secondOpIters.size())
      << "the length of input factors does not match second Ops private iters";
  auto findPos = [&](IterVar iv) {
    int idx = 0;
    for (auto i : _secondOpIters) {
      if (i == iv)
        return idx;
      idx++;
    }
    return -1;
  };
  bool splited = secondOpIters.size() > _secondOpIters.size();
  if (splited) {
    for (auto i : splitRelations) {
      int pos = findPos(i->parent);
      if (pos >= 0)
        i->inner->ext = factors[pos];
    }
    return;
  }
  Array<IterVar> outer, inner;
  for (size_t i = 0; i < factors.size(); i++) {
    IterVar parent = secondOpIters[i];
    CHECK(factors[i] <= parent->ext) << "tiling factor larger than axis extent";
    IterVar outer_, inner_;
    outer_ = IterVar(parent->index, -1, parent->iv_type,
                     parent->name.copy_with_suffix(".outer"), tir::Var());
    // the inner iter success the Var
    inner_ =
        IterVar(parent->index, factors[findPos(parent)], parent->iv_type,
                parent->name.copy_with_suffix(".inner"), parent->originVar);
    outer.push_back(outer_);
    inner.push_back(inner_);
    splitRelations.push_back(Split(parent, outer_, inner_));
    for (size_t j = 0; j < shareRelations.size(); j++)
      if (shareRelations[j]->lower == parent) {
        shareRelations[j]->lower = outer_;
      }
  }
  outer.insert(outer.end(), inner.begin(), inner.end());
  secondOpIters = outer;
}

void IterGraphNode::setFirstOpPermute(Array<IntImm> _permutation) {
  std::vector<size_t> permutation;
  for (auto i : _permutation)
    permutation.push_back(i->value);
  std::vector<size_t> permutation_ = permutation;
  std::sort(permutation_.begin(), permutation_.end());
  CHECK(permutation_.size() == _firstOpIters.size())
      << "setPermute takes Sigma([len(firstOpIters)]) as input.";
  for (size_t i = 0; i < permutation_.size(); i++)
    CHECK(permutation_[i] == i)
        << "setPermute takes Sigma([len(firstOpIters)]) as input.";
  auto findIvInIters = [&](IterVar iv) {
    for (auto i : splitRelations)
      if (i->parent == iv)
        return i->outer;
    return iv;
  };

  Array<IterVar> newOpList;
  for (size_t i = 0; i < _firstOpIters.size(); i++) {
    IterVar iv = findIvInIters(_firstOpIters[permutation[i]]);
    newOpList.push_back(iv);
  }
  for (size_t i = _firstOpIters.size(); i < firstOpIters.size(); i++)
    newOpList.push_back(firstOpIters[i]);
  firstOpIters = newOpList;
}

void IterGraphNode::setSecondOpPermute(Array<IntImm> _permutation) {
  std::vector<size_t> permutation;
  for (auto i : _permutation)
    permutation.push_back(i->value);
  std::vector<size_t> permutation_ = permutation;
  std::sort(permutation_.begin(), permutation_.end());
  CHECK(permutation_.size() == _secondOpIters.size())
      << "setPermute takes Sigma([len(secondOpIters)]) as input.";
  for (size_t i = 0; i < permutation_.size(); i++)
    CHECK(permutation_[i] == i)
        << "setPermute takes Sigma([len(secondOpIters)]) as input.";

  auto findIvInIters = [&](IterVar iv) {
    for (auto i : splitRelations)
      if (i->parent == iv)
        return i->outer;
    return iv;
  };

  Array<IterVar> newOpList;
  for (size_t i = 0; i < _secondOpIters.size(); i++) {
    IterVar iv = findIvInIters(_secondOpIters[permutation[i]]);
    newOpList.push_back(iv);
  }
  for (size_t i = _secondOpIters.size(); i < secondOpIters.size(); i++)
    newOpList.push_back(secondOpIters[i]);
  secondOpIters = newOpList;
}

void IterGraphNode::setAttach(size_t attach_pos) {
  CHECK(attach_pos <= _secondOpIters.size() && attach_pos >= 0)
      << "attach pos should be in range [0, len(secondOpIters)]";
  attachPos = attach_pos;
}

void IterGraphNode::setFusion(FusionItem fusionItem) {
  setFirstOpTiling(fusionItem->firstOpTiling->factors);
  setSecondOpTiling(fusionItem->secondOpTiling->factors);
  setFirstOpPermute(fusionItem->firstOpPermute->permute);
  setSecondOpPermute(fusionItem->secondOpPermute->permute);
  setAttach(fusionItem->attachPos->attachPos);
  applyAll();
}

FusionSpace IterGraphNode::getSearchSpace() {
  Array<SearchSpace> ret;
  auto getExtents = [&](Array<IterVar> ivs) {
    Array<IntImm> ret;
    for (auto iv : ivs)
      ret.push_back(IntImm(DataType::Int(32), iv->ext));
    return ret;
  };
  return FusionSpace(PermuteSpace(_firstOpIters.size()),
                     PermuteSpace(_secondOpIters.size()),
                     TilingSpace(getExtents(_firstOpIters)),
                     TilingSpace(getExtents(_secondOpIters)),
                     AttachSpace(_secondOpIters.size()));
}

void IterGraphNode::applyAll() {
  // 0. check if both ops are splited, clear shared bit
  CHECK(firstOpIters.size() > _firstOpIters.size())
      << "split op1 before applyAll";
  CHECK(secondOpIters.size() > _secondOpIters.size())
      << "split op 2before applyAll";
  for (auto iv : firstOpIters)
    iv->shared = false;
  for (auto iv : secondOpIters)
    iv->shared = false;

  // 1. set Common Iters
  Array<IterVar> commonLoops_;
  for (size_t i = 0; i < attachPos; i++)
    commonLoops_.push_back(secondOpIters[i]);
  commonIters = commonLoops_;

  // 2. find the shared axis & common axis and replace the axis in shared
  // relation.
  auto findSplitPos = [&](IterVar iv) {
    for (auto i : splitRelations)
      if (i->outer == iv)
        return i;
    throw Error("findSplitPos error");
  };
  for (auto iv : firstOpIters) {
    for (auto common : commonIters) {
      for (auto share : shareRelations) {
        if (share->upper == iv && share->lower == common) {
          // update the split relation for iv
          iv->shared = common->shared = true;
          auto split_op1 = findSplitPos(share->upper);
          auto split_op2 = findSplitPos(share->lower);
          split_op1->inner->ext = split_op2->inner->ext;
        }
      }
    }
  }

  // 3. set the extent for outer loops
  for (auto split : splitRelations) {
    IterVar outer = split->outer;
    CHECK(split->parent->ext >= 0 && split->inner->ext >= 0)
        << "the split parent or inner's extent hasn't set";
    split->outer->ext =
        (split->parent->ext + split->inner->ext - 1) / split->inner->ext;
  }
}

Map<tir::Var, IntImm> IterGraphNode::inferBound() const {
  Map<tir::Var, IntImm> bounds;
  auto is_common = [&](IterVar iv) {
    for (auto common : commonIters) {
      if (iv == common)
        return true;
      for (auto share : shareRelations) {
        if (share->lower == common && share->upper == iv)
          return true;
      }
    }
    return false;
  };
  for (auto split : splitRelations) {
    if (is_common(split->outer))
      bounds.Set(split->inner->originVar,
                 IntImm(DataType::Int(32), split->inner->ext));
    else
      bounds.Set(split->inner->originVar,
                 IntImm(DataType::Int(32), split->parent->ext));
  }
  return bounds;
}

int IterGraphNode::getNumOfBlocks() const {
  int ret = 1;
  for (auto iv : commonIters) {
    CHECK(iv->ext >= 0)
        << "the extent for outerLoops is unset. Have you called applyAll?";
    ret *= iv->ext;
  }
  return ret;
}

int IterGraphNode::getParallelism() const {
  int ret = 1;
  for (auto iv : commonIters) {
    if (iv->iv_type != IV_Type::SPATIAL)
      continue;
    ret *= iv->ext;
  }
  return ret;
}

int IterGraphNode::getRedundancy() const {
  int ret = 1;
  for (auto iv : commonIters)
    if (!iv->shared)
      ret *= iv->ext;
  return ret;
}

/*! \brief get the first Op's memvisit*/
int IterGraphNode::getFirstOpDataVolume() const {
  int n_block = getNumOfBlocks();
  int fp = 0;
  for (auto acf : firstOpReadAccessFuncs)
    for (auto fp_ : acf->getFootprint(bounds))
      fp += fp_;
  // the write of firstOp is on chip
  // for (auto fp_ : firstOpWriteAccessFunc->getFootprint(bounds))
  //   fp += fp_;
  return fp * n_block;
}

/*! \brief get the first Op's workload*/
int IterGraphNode::getFirstOpWorkload() const {
  int n_block = getNumOfBlocks();
  int fp = firstOpWriteAccessFunc->getWorkload(bounds);
  return fp * n_block;
}

/*! \brief get the first Op's blockSize*/
int IterGraphNode::getFirstOpBufferSize() const {
  int fp = 0;
  for (auto acf : firstOpReadAccessFuncs)
    for (auto fp_ : acf->getFootprint(bounds))
      fp += fp_;
  for (auto fp_ : firstOpWriteAccessFunc->getFootprint(bounds))
    fp += fp_;
  return fp;
}

/*! \brief get the second Op's memvisit*/
int IterGraphNode::getSecondOpDataVolume() const {
  int n_block = getNumOfBlocks();
  int fp = 0;
  for (auto acf : secondOpReadAccessFuncs)
    for (auto fp_ : acf->getFootprint(bounds))
      fp += fp_;
  for (auto fp_ : secondOpWriteAccessFunc->getFootprint(bounds))
    fp += fp_;
  // the read of first op's output does not cause data move
  for (auto fp_ : firstOpWriteAccessFunc->getFootprint(bounds))
    fp -= fp_;
  return fp * n_block;
}
/*! \brief get the second Op's workload*/
int IterGraphNode::getSecondOpWorkload() const {
  int n_block = getNumOfBlocks();
  int fp = secondOpWriteAccessFunc->getWorkload(bounds);
  return fp * n_block;
}
/*! \brief get the second Op's blockSize*/
int IterGraphNode::getSecondOpBufferSize(bool writeThrough) const {
  int fp = 0;
  for (auto acf : secondOpReadAccessFuncs)
    for (auto fp_ : acf->getFootprint(bounds))
      fp += fp_;
  if (!writeThrough)
    for (auto fp_ : secondOpWriteAccessFunc->getFootprint(bounds))
      fp += fp_;
  return fp;
}

void IterGraphNode::writeResult(FusionResult res) {
  if (!resultPath->size)
    return;
  std::ofstream outfile;
  outfile.open(std::string(resultPath),
               std::ios_base::app); // append instead of overwrite
  outfile << "{";
  outfile << "outerIter:";
  outfile << "[";
  for (auto iter : commonIters) {
    outfile << "{";
    for (auto split : splitRelations)
      if (split->outer == iter) {
        outfile << "name: " << split->parent->name << ", ";
        outfile << "factor: " << split->inner->ext;
      }
    outfile << " }, ";
  }
  outfile << "],";
  outfile << "locality: " << res->locality << ", ";
  outfile << "parallelism: " << res->parallelism << ", ";
  outfile << "redundancy: " << res->redundancy << ", ";
  outfile << "valid: " << res->valid << ", ";
  outfile << "}\n";
  outfile.close();
}

/*! \brief get the analytical result */
FusionResult
IterGraphNode::getAnalyticalResult(hardware::HardwareParam hw_param,
                                   int bytePerEle, bool writeThrough) {
  applyAll();
  bounds = inferBound();
  int n_blocks = getNumOfBlocks();
  int firstOpDataVolume = getFirstOpDataVolume() * bytePerEle;
  int firstOpBufferSize = getFirstOpBufferSize() * bytePerEle;
  int firstOpWorkload = getFirstOpWorkload();
  int secondOpDataVolume = getSecondOpDataVolume() * bytePerEle;
  int secondOpBufferSize = getSecondOpBufferSize(writeThrough) * bytePerEle;
  int secondOpWorkload = getSecondOpWorkload();
  int memUse = (firstOpBufferSize + secondOpBufferSize);
  bool valid = (memUse) <= (hw_param->shared_memory_per_group_kb * 1000);
  double locality =
      valid ? 1 / (double)(firstOpDataVolume + secondOpDataVolume) : -INFINITY;
  int parallelism = std::min(
      std::min(getParallelism(),
               hw_param->num_groups * hw_param->num_processors_per_group),
      hw_param->num_groups *
          ((int)hw_param->shared_memory_per_group_kb * 1000 / memUse));
  int redundancy = getRedundancy();
  FusionResult res = FusionResult(
      bounds, firstOpDataVolume, firstOpWorkload, firstOpBufferSize,
      secondOpDataVolume, secondOpWorkload, secondOpBufferSize, locality,
      parallelism, redundancy, n_blocks, valid);
  writeResult(res);
  return res;
}

/*! \brief looplike lightweight visualize */
void IterGraphNode::visualize() {
  std::cout << "-----IterGraph Visualization------\n";
  applyAll();
  size_t n_tab = 0;
  for (auto common : commonIters) {
    for (size_t i = 0; i < n_tab; i++)
      std::cout << "\t";
    std::cout << "for  " << common->name << " in [0, " << common->ext << "):";
    if (common->shared)
      std::cout << "\"shared\"";
    std::cout << "\n";
    n_tab++;
  }
  size_t n_tab_ = n_tab;
  for (auto iv : firstOpIters) {
    if (iv->shared)
      continue;
    for (size_t i = 0; i < n_tab_; i++)
      std::cout << "\t";
    std::cout << "for  " << iv->name << " in [0, " << iv->ext << "):\n";
    n_tab_ += 1;
  }
  n_tab_ = n_tab;
  for (size_t _ = attachPos; _ < secondOpIters.size(); _++) {
    auto iv = secondOpIters[_];
    for (size_t i = 0; i < n_tab_; i++)
      std::cout << "\t";
    std::cout << "for  " << iv->name << " in [0, " << iv->ext << "):\n";
    n_tab_++;
  }
  std::cout << "---------------------------------\n";
}

inline IterGraph buildIterGraph(SerialFusionState sfState, String path) {
  OpHyperState ops1, ops2;
  std::tie(ops1, ops2) = sfState->getCubicOpPair();
  Array<IterVar> firstOpIters = ops1->getAllIters();
  Array<IterVar> secondOpIters = ops2->getAllIters();
  Array<Array<tir::IterVar>> shared_axis =
      share_axis_analysis(ops1->op, ops2->op);
  std::unordered_map<tir::IterVar, IterVar> IterMap = ops1->getIterMap();
  IterMap.insert(ops2->getIterMap().begin(), ops2->getIterMap().end());
  Array<Share> sharedIterPairs;
  for (auto sharePair : shared_axis) {
    CHECK(IterMap.count(sharePair[0])) << sharePair[0] << " not in the IterMap";
    CHECK(IterMap.count(sharePair[1])) << sharePair[1] << " not in the IterMap";
    sharedIterPairs.push_back(
        Share(IterMap.at(sharePair[0]), IterMap.at(sharePair[1])));
  }
  Array<AccessFunction> firstOpReadAccessFunction = ops1->ReadAccessFunctions();
  Array<AccessFunction> secondOpReadAccessFunction =
      ops2->ReadAccessFunctions();
  AccessFunction firstOpWriteAccessFunc = ops1->WriteAccessFunctions();
  AccessFunction secondOpWriteAccessFunc = ops2->WriteAccessFunctions();
  // std::cout << "op1" << std::endl;
  // std::cout << ops1->op << std::endl;
  // std::cout << "op2" << std::endl;
  // std::cout << ops2->op << std::endl;
  int readProducerPos = ops2->getFirstProducerPos();
  CHECK(readProducerPos >= 0)
      << readProducerPos << " Can't find a producer for the second op.";
  return IterGraph(firstOpIters, secondOpIters, sharedIterPairs,
                   firstOpReadAccessFunction, secondOpReadAccessFunction,
                   firstOpWriteAccessFunc, secondOpWriteAccessFunc,
                   readProducerPos, ops1->op, ops2->op, path);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterGraphNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const IterGraphNode *>(node.get());
      p->PrintIndent();
      p->stream << "---------------IterGraph------------------\n";
      p->stream << "op1: " << op->op1->name;
      p->stream << ", op2: " << op->op2->name;
      p->stream << ",\n";
      p->stream << "_firstOpIters:\t";
      p->Print(op->_firstOpIters);
      p->stream << ",\n";
      p->stream << "_secondOpIters:\t";
      p->Print(op->_secondOpIters);
      p->stream << "\n------------------------------------------";
    });

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.build_iter_graph")
    .set_body_typed(buildIterGraph);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setAttach")
    .set_body_method<IterGraph>(&IterGraphNode::setAttach);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setFirstOpTiling")
    .set_body_method<IterGraph>(&IterGraphNode::setFirstOpTiling);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setSecondOpTiling")
    .set_body_method<IterGraph>(&IterGraphNode::setSecondOpTiling);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setFirstOpPermute")
    .set_body_method<IterGraph>(&IterGraphNode::setFirstOpPermute);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setSecondOpPermute")
    .set_body_method<IterGraph>(&IterGraphNode::setSecondOpPermute);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.applyAll")
    .set_body_method<IterGraph>(&IterGraphNode::applyAll);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setFusion")
    .set_body_method<IterGraph>(&IterGraphNode::setFusion);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.display")
    .set_body_method<IterGraph>(&IterGraphNode::visualize);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getAnalyticalResult")
    .set_body_method<IterGraph>(&IterGraphNode::getAnalyticalResult);
} // namespace auto_tensorize

} // namespace ditto