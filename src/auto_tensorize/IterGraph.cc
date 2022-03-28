#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/evaluator.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/pattern.h>
#include <auto_tensorize/state.h>
#include <fstream>
#include <sys/stat.h>
#include <utils/iter_domain.h>
namespace ditto
{

  namespace auto_tensorize
  {

    Share::Share(IterVar upper, IterVar lower)
    {
      auto n = make_object<ShareNode>();
      n->upper = upper;
      n->lower = lower;
      data_ = std::move(n);
    }
    TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<ShareNode>([](const ObjectRef &node, ReprPrinter *p)
                                 {
      auto *op = static_cast<const ShareNode *>(node.get());
      p->PrintIndent();
      p->stream << "Share(";
      p->stream << "upper: ";
      p->Print(op->upper);
      p->stream << ", ";
      p->stream << "lower: ";
      p->Print(op->lower);
      p->stream << ") "; });
    Split::Split(IterVar parent, IterVar outer, IterVar inner)
    {
      auto n = make_object<SplitNode>();
      n->parent = parent;
      n->outer = outer;
      n->inner = inner;
      data_ = std::move(n);
    }
    TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<SplitNode>([](const ObjectRef &node, ReprPrinter *p)
                                 {
      auto *op = static_cast<const SplitNode *>(node.get());
      p->PrintIndent();
      p->stream << "Split(";
      p->stream << "outer: ";
      p->Print(op->outer);
      p->stream << ", ";
      p->stream << "inner: ";
      p->Print(op->inner);
      p->stream << ") "; });
    IterVar::IterVar(int idx_, FACTOR ext_, IV_Type iv_type_, tvm::tir::Var name_,
                     tvm::tir::Var originVar)
    {
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
        .set_dispatch<IterVarNode>([](const ObjectRef &node, ReprPrinter *p)
                                   {
      auto *op = static_cast<const IterVarNode *>(node.get());
      std::string classes[3] = {"S1", "S2", "R"};
      p->PrintIndent();
      // p->stream << "IterVar(";
      // p->stream << "ext: " << op->ext << ", ";
      // p->stream << "type: " << (op->iv_type == IV_Type::SPATIAL? "S":"R") <<
      // ", ";
      p->stream << op->name << " ";
      switch (op->iv_type) {
      case IV_Type::FIRSTSPATIAL:
        p->stream << "S1";
        break;
      case IV_Type::SECONDSPATIAL:
        p->stream << "S2";
        break;
      case IV_Type::REDUCE:
        p->stream << "R";
        break;
      case IV_Type::TENSORIZE:
        p->stream << "T";
        break;
      default:
        p->stream << "U";
      }
      p->stream << " " << op->ext; });

    AccessFunction::AccessFunction(te::Operation op,
                                   Array<Array<PrimExpr>> access_indices,
                                   Array<tir::Var> absentVars,
                                   Array<tir::Var> presentVars)
    {
      auto node = make_object<AccessFunctionNode>();
      node->op = op;
      node->access_indices = access_indices;
      node->absentVars = absentVars;
      node->presentVars = presentVars;
      data_ = node;
    }
    // TODO: the footprint of non-cartesian product
    std::vector<int>
    AccessFunctionNode::getFootprint(Map<tir::Var, IntImm> bounds)
    {
      int idx = 0;
      Map<tir::Var, Range> ori_ranges;
      for (auto kv : bounds)
        ori_ranges.Set(kv.first, Range(0, kv.second));
      std::vector<int> ret;
      for (auto access_index : access_indices)
      {
        Map<tir::Var, PrimExpr> vars_to_infer;
        for (auto index : access_index)
        {
          vars_to_infer.Set(tvm::tir::Var("v" + std::to_string(idx++)), index);
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
    /*! \brief get the product of absent axis*/
    int AccessFunctionNode::getProductOfAbsentVars(Map<tir::Var, IntImm> bounds)
    {
      int value = 1;
      for (auto var : absentVars)
      {
        CHECK(bounds.count(var)) << var << " dosen't appear in bounds" << std::endl;
        value *= bounds[var]->value;
      }
      return value;
    }

    void AccessFunctionNode::repalceVars(Map<tir::Var, tir::Var> map)
    {
      Array<Array<PrimExpr>> new_indices;
      for (auto arr : access_indices)
      {
        Array<PrimExpr> tmp;
        for (auto expr : arr)
        {
          tmp.push_back(utils::ReplaceVars(expr, map));
        }
        new_indices.push_back(tmp);
      }
      access_indices = new_indices;
      Array<tir::Var> newAbsentVars, newPresentVars;
      for (auto &var : absentVars)
      {
        if (map.count(var))
          newAbsentVars.push_back(map.at(var));
        else
          newAbsentVars.push_back(var);
      }
      for (auto &var : presentVars)
      {
        if (map.count(var))
          newPresentVars.push_back(map.at(var));
        else
          newPresentVars.push_back(var);
      }
      absentVars = newAbsentVars;
      presentVars = newPresentVars;
    }

    void AccessFunctionNode::setAbsentVars(Array<tir::Var> absentVars_)
    {
      absentVars = absentVars_;
      return;
    }

    int AccessFunctionNode::getWorkload(Map<tir::Var, IntImm> bounds)
    {
      auto cop = op.as<te::ComputeOpNode>();
      int ret = 1;
      for (auto iv : cop->axis)
      {
        tir::Var var = iv->var;
        ret *= bounds[var]->value;
      }
      for (auto iv : cop->reduce_axis)
      {
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
                         Array<IterVar> firstOpTensorizeIters,
                         Array<IterVar> secondOpTensorizeIters)
    {
      auto n = make_object<IterGraphNode>();
      n->configSet = false;
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
      Map<tir::Var, IntImm> bounds;

      n->firstOpTensorizeIters = firstOpTensorizeIters;
      n->secondOpTensorizeIters = secondOpTensorizeIters;
      n->setTotalFp();
      data_ = std::move(n);
    }

    void IterGraphNode::setTotalFp()
    {
      for (auto iv : firstOpTensorizeIters)
      {
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      }
      for (auto iv : secondOpTensorizeIters)
      {
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      }

      for (auto iv : _firstOpIters)
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));

      for (auto iv : _secondOpIters)
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));

      totalFp = getFirstOpBufferSize() + getSecondOpBufferSize();
    }

    void IterGraphNode::setFirstOpTiling(Array<IntImm> factors_)
    {
      std::vector<FACTOR> factors;
      for (auto i : factors_)
        factors.push_back(i->value);
      CHECK(factors.size() == _firstOpIters.size())
          << "the length of input factors does not match first Ops private iters";
      auto findPos = [&](IterVar iv)
      {
        int idx = 0;
        for (auto i : _firstOpIters)
        {
          if (i == iv)
            return idx;
          idx++;
        }
        return -1;
      };
      bool splited = firstOpIters.size() > _firstOpIters.size();
      if (splited)
      {
        for (auto i : splitRelations)
        {
          int pos = findPos(i->parent);
          if (pos >= 0)
            i->inner->ext = factors[pos];
        }
        return;
      }
      Array<IterVar> outer, inner;
      for (size_t i = 0; i < factors.size(); i++)
      {
        IterVar parent = _firstOpIters[i];
        CHECK(factors[i] <= parent->ext) << "tiling factor " << i << "," << factors[i] << " larger than axis extent for " << parent;
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
          if (shareRelations[j]->upper == parent)
          {
            shareRelations[j]->upper = outer_;
          }
      }
      outer.insert(outer.end(), inner.begin(), inner.end());
      firstOpIters = outer;
    }

    void IterGraphNode::setSecondOpTiling(Array<IntImm> factors_)
    {
      std::vector<FACTOR> factors;
      for (auto i : factors_)
        factors.push_back(i->value);
      CHECK(factors.size() == _secondOpIters.size())
          << "the length of input factors does not match second Ops private iters";
      auto findPos = [&](IterVar iv)
      {
        int idx = 0;
        for (auto i : _secondOpIters)
        {
          if (i == iv)
            return idx;
          idx++;
        }
        return -1;
      };
      bool splited = secondOpIters.size() > _secondOpIters.size();
      if (splited)
      {
        for (auto i : splitRelations)
        {
          int pos = findPos(i->parent);
          if (pos >= 0)
            i->inner->ext = factors[pos];
        }
        return;
      }
      Array<IterVar> outer, inner;
      for (size_t i = 0; i < factors.size(); i++)
      {
        IterVar parent = _secondOpIters[i];
        CHECK(factors[i] <= parent->ext)
            << "tiling factor larger than axis extent. iv: " << _secondOpIters[i]
            << " " << factors[i] << " > " << parent->ext;
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
          if (shareRelations[j]->lower == parent)
          {
            shareRelations[j]->lower = outer_;
          }
      }
      outer.insert(outer.end(), inner.begin(), inner.end());
      secondOpIters = outer;
    }

    void IterGraphNode::setFirstOpPermute(Array<IntImm> _permutation)
    {
      std::vector<size_t> permutation;
      for (auto i : _permutation)
        permutation.push_back(i->value);
      std::vector<size_t> permutation_ = permutation;
      std::sort(permutation_.begin(), permutation_.end());
      CHECK(permutation_.size() == _firstOpIters.size())
          << "setPermute takes sigma([len(firstOpIters)]) as input.";
      for (size_t i = 0; i < permutation_.size(); i++)
        CHECK(permutation_[i] == i)
            << "setPermute takes sigma([len(firstOpIters)]) as input. The permute "
               "given is "
            << _permutation;
      auto findIvInIters = [&](IterVar iv)
      {
        for (auto i : splitRelations)
          if (i->parent == iv)
            return i->outer;
        return iv;
      };

      Array<IterVar> newOpList;
      for (size_t i = 0; i < _firstOpIters.size(); i++)
      {
        IterVar iv = findIvInIters(_firstOpIters[permutation[i]]);
        newOpList.push_back(iv);
      }
      for (size_t i = _firstOpIters.size(); i < firstOpIters.size(); i++)
        newOpList.push_back(firstOpIters[i]);
      firstOpIters = newOpList;
    }

    void IterGraphNode::setSecondOpPermute(Array<IntImm> _permutation)
    {
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

      auto findIvInIters = [&](IterVar iv)
      {
        for (auto i : splitRelations)
          if (i->parent == iv)
            return i->outer;
        return iv;
      };

      Array<IterVar> newOpList;
      for (size_t i = 0; i < _secondOpIters.size(); i++)
      {
        IterVar iv = findIvInIters(_secondOpIters[permutation[i]]);
        newOpList.push_back(iv);
      }
      for (size_t i = _secondOpIters.size(); i < secondOpIters.size(); i++)
        newOpList.push_back(secondOpIters[i]);
      secondOpIters = newOpList;
    }

    void IterGraphNode::setAttach(size_t attach_pos)
    {
      CHECK(attach_pos <= _secondOpIters.size() && attach_pos >= 0)
          << "attach pos should be in range [0, len(secondOpIters)]";
      attachPos = attach_pos;
    }

    void IterGraphNode::setFusion(FusionItem fusionItem)
    {
      setFirstOpTiling(fusionItem->firstOpTiling->factors);
      setSecondOpTiling(fusionItem->secondOpTiling->factors);
      setFirstOpPermute(fusionItem->firstOpPermute->permute);
      setSecondOpPermute(fusionItem->secondOpPermute->permute);
      setAttach(fusionItem->attachPos->attachPos);
      setFusionLevel(fusionItem->fusionLevel);
      applyAll();
    }

    FusionSpace IterGraphNode::getSearchSpace()
    {
      Array<SearchSpace> ret;
      auto getExtents = [&](Array<IterVar> ivs)
      {
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

    void IterGraphNode::applyAll()
    {
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
      auto findSplitPos = [&](IterVar iv)
      {
        for (auto i : splitRelations)
          if (i->outer == iv)
            return i;
        throw Error("findSplitPos error");
      };
      for (auto iv : firstOpIters)
      {
        for (auto common : commonIters)
        {
          for (auto share : shareRelations)
          {
            if (share->upper == iv && share->lower == common)
            {
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
      for (auto split : splitRelations)
      {
        IterVar outer = split->outer;
        CHECK(split->parent->ext >= 0 && split->inner->ext >= 0)
            << "the split parent or inner's extent hasn't set";
        split->outer->ext =
            (split->parent->ext + split->inner->ext - 1) / split->inner->ext;
      }
    }

    Map<tir::Var, IntImm> IterGraphNode::inferBound() const
    {
      Map<tir::Var, IntImm> bounds;
      for (auto iv : firstOpTensorizeIters)
      {
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      }
      for (auto iv : secondOpTensorizeIters)
      {
        bounds.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      }
      auto is_common = [&](IterVar iv)
      {
        for (auto common : commonIters)
        {
          if (iv == common)
            return true;
          for (auto share : shareRelations)
          {
            if (share->lower == common && share->upper == iv)
              return true;
          }
        }
        return false;
      };
      for (auto split : splitRelations)
      {
        if (is_common(split->outer))
          bounds.Set(split->inner->originVar,
                     IntImm(DataType::Int(32), split->inner->ext));
        else
          bounds.Set(split->inner->originVar,
                     IntImm(DataType::Int(32), split->parent->ext));
      }
      return bounds;
    }

    double IterGraphNode::getNumOfBlocks() const
    {
      double ret = 1;
      for (auto iv : commonIters)
      {
        CHECK(iv->ext >= 0)
            << "the extent for outerLoops is unset. Have you called applyAll?";
        ret *= iv->ext;
      }
      return ret;
    }

    double IterGraphNode::getParallelism() const
    {
      CHECK(parallelism_ > 0);
      double ret = 1;
      for (auto iv : commonIters)
      {
        if (iv->iv_type == IV_Type::REDUCE)
          continue;
        ret *= iv->ext;
      }
      return std::min(ret, parallelism_);
    }

    double IterGraphNode::getRedundancy() const
    {
      double ret = 1;
      for (auto iv : commonIters)
        if (!iv->shared)
          ret *= iv->ext;
      return ret;
    }

    /*! \brief get the first Op's memvisit*/
    double IterGraphNode::getFirstOpDataVolume() const
    {
      CHECK(tensorWeight.size() == 2);
      double fp = 0;
      for (auto acf : firstOpReadAccessFuncs)
        for (auto fp_ : acf->getFootprint(bounds))
        {
          fp += fp_ * tensorWeight[0];
        }
      return fp * bytePerEle * _outerCost;
    }

    /*! \brief get the first Op's workload*/
    double IterGraphNode::getFirstOpWorkload() const
    {
      double n_block = getNumOfBlocks();
      double fp = firstOpWriteAccessFunc->getWorkload(bounds);
      return fp * n_block;
    }

    /*! \brief get the first Op's blockSize*/
    double IterGraphNode::getFirstOpBufferSize(bool considerWrite) const
    {
      double fp = 0;
      for (auto acf : firstOpReadAccessFuncs)
        for (auto fp_ : acf->getFootprint(bounds))
          fp += fp_;
      if (considerWrite)
        for (auto fp_ : firstOpWriteAccessFunc->getFootprint(bounds))
          fp += fp_;
      return fp * bytePerEle;
    }

    /*! \brief get the second Op's memvisit*/
    double IterGraphNode::getSecondOpDataVolume() const
    {
      CHECK(tensorWeight.size() == 2);
      double fp = 0;
      for (auto acf : secondOpReadAccessFuncs)
        for (auto fp_ : acf->getFootprint(bounds))
        {
          fp += fp_ * tensorWeight[0];
        }
      for (auto fp_ : secondOpWriteAccessFunc->getFootprint(bounds))
      {
        fp += fp_ * tensorWeight[1];
      }
      // the read of first op's output does not cause data move
      for (auto fp_ : firstOpWriteAccessFunc->getFootprint(bounds))
      {
        fp -= fp_ * tensorWeight[0];
      }
      return fp * bytePerEle * _outerCost;
    }
    /*! \brief get the second Op's workload*/
    double IterGraphNode::getSecondOpWorkload() const
    {
      double n_block = getNumOfBlocks();
      double fp = secondOpWriteAccessFunc->getWorkload(bounds);
      return fp * n_block;
    }
    /*! \brief get the second Op's blockSize*/
    double IterGraphNode::getSecondOpBufferSize() const
    {
      double fp = 0;
      for (auto acf : secondOpReadAccessFuncs)
        for (auto fp_ : acf->getFootprint(bounds))
          fp += fp_;
      if (!writeThrough)
        for (auto fp_ : secondOpWriteAccessFunc->getFootprint(bounds))
          fp += fp_;
      return fp * bytePerEle;
    }

    /*! \brief get the problem size after parallel */
    Map<tir::Var, IntImm> getPbsz();

    /*! \brief get the analytical result */
    FusionResult
    IterGraphNode::getAnalyticalResult()
    {
      CHECK(configSet) << "please set config first";
      applyAll();
      bounds = inferBound();
      scheduleParallel();
      double cacheSize = cacheSizes[fusionLevel];
      double n_blocks = getNumOfBlocks();
      double firstOpDataVolume = getFirstOpDataVolume();
      double firstOpBufferSize = getFirstOpBufferSize(true);
      double firstOpWorkload = getFirstOpWorkload();
      double secondOpDataVolume = getSecondOpDataVolume();
      double secondOpBufferSize = getSecondOpBufferSize();
      double secondOpWorkload = getSecondOpWorkload();
      double memUse = std::max(firstOpBufferSize, secondOpBufferSize);
      bool valid = (memUse) <= cacheSize;
      double locality =
          valid ? 1 / (double)(firstOpDataVolume + secondOpDataVolume) : -INFINITY;
      double parallelism =
          std::min(getParallelism(), parallelism_ * (cacheSize / memUse)) / _maxThreadIter;
      double redundancy = getRedundancy();
      FusionResult res =
          FusionResult(bounds, firstOpDataVolume, firstOpWorkload,
                       firstOpBufferSize, secondOpDataVolume, secondOpWorkload,
                       secondOpBufferSize, locality, parallelism, redundancy,
                       n_blocks, valid, fusionLevel, bytePerEle, cacheSize, memUse);
      return res;
    }

    void IterGraphNode::setFusionLevel(int fusionLevel_)
    {
      bool found = false;
      for (auto l : fusionLevels)
      {
        if (l == fusionLevel_)
          found = true;
      }
      CHECK(found) << "have you set config? ";
      fusionLevel = fusionLevel_;
      return;
    }

    void IterGraphNode::scheduleParallel()
    {
      std::unordered_map<int, IntImm> parallelSchedule;
      std::unordered_map<int, int> idx2ext;
      std::vector<int> parallelIndices;
      int initCost = 1;
      for (auto iv : commonIters)
      {
        if (iv->iv_type == IV_Type::TENSORIZE)
          continue;
        if (iv->iv_type == IV_Type::REDUCE )
          initCost *= iv->ext;
        idx2ext[iv->index] = iv->ext;
        parallelIndices.push_back(iv->index);
      }
      int bestCost = 1e9;
      int bestMaxThreadIter;
      const int totalThread = (int)parallelism_;
      int exts[parallelIndices.size()];
      
      std::function<void(int, int, int)> func = [&](size_t __idx, int cost, int usedThread)
      {
        int idx = parallelIndices[__idx];
        if (__idx == parallelIndices.size())
        {
          cost *= ((totalThread + usedThread - 1) / (totalThread));
          if (cost < bestCost)
          {
            bestCost = cost;
            bestMaxThreadIter = (totalThread + usedThread - 1) / (totalThread);
            for (size_t i = 0; i < parallelIndices.size(); i++)
            {
              parallelSchedule[parallelIndices[i]] = IntImm(DataType::Int(32), exts[i]);
            }
          }
          return;
        }
        int ext = idx2ext[idx];
        if (cost * ((usedThread + totalThread - 1) / (totalThread)) >= bestCost)
          return;
        for (int i = ext; i >= 1; i--)
        {
          exts[__idx] = i;
          func(__idx + 1, cost * ((ext + i - 1) / i), usedThread * i);
        }
      };
      func(0, initCost, 1);
      Map<tir::Var, IntImm> boundsAfterParallel;
      for (auto iv : firstOpTensorizeIters)
        boundsAfterParallel.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      for (auto iv : secondOpTensorizeIters)
        boundsAfterParallel.Set(iv->originVar, IntImm(DataType::Int(32), iv->ext));
      for (auto iv : _firstOpIters)
      {
        int ext = iv->ext;
        if (parallelSchedule.count(iv->index))
        {
          int tmp = parallelSchedule.at(iv->index)->value;
          ext = (ext + tmp - 1) / tmp;
        }
        boundsAfterParallel.Set(iv->originVar, IntImm(DataType::Int(32), ext));
      }
      for (auto iv : _secondOpIters)
      {
        int ext = iv->ext;
        if (parallelSchedule.count(iv->index))
        {
          int tmp = parallelSchedule.at(iv->index)->value;
          ext = (ext + tmp - 1) / tmp;
        }
        boundsAfterParallel.Set(iv->originVar, IntImm(DataType::Int(32), ext));
      }
      _maxThreadIter = bestMaxThreadIter;
      _outerCost = bestCost;
      _parallelSchedule = parallelSchedule;
      _boundsAfterParallel = boundsAfterParallel;
      return;
    }

    std::pair<bool, double> IterGraphNode::getCost(double *occupancy, double *parallelism_p, double *memUse_p)
    {
      applyAll();
      bounds = inferBound();
      scheduleParallel();
      CHECK(fusionLevel >= 0);
      double cacheSize = cacheSizes[fusionLevel];
      double firstOpDataVolume = getFirstOpDataVolume();
      double firstOpBufferSize = getFirstOpBufferSize(true);
      double secondOpDataVolume = getSecondOpDataVolume();
      double secondOpBufferSize = getSecondOpBufferSize();
      double memUse = std::max(firstOpBufferSize, secondOpBufferSize);
      bool valid = (memUse) <= cacheSize;

      double parallelism =
          std::min(getParallelism(), parallelism_ * (cacheSize / memUse)) / _maxThreadIter;
      if (occupancy){
        *occupancy = memUse / cacheSize;
      }
      if (parallelism_p)
        *parallelism_p = parallelism;
      if (memUse_p)
        *memUse_p = memUse;
      return {valid,
              (firstOpDataVolume + secondOpDataVolume) / (cacheBandwidth[fusionLevel])};
    }

    /*! \brief looplike lightweight visualize */
    void IterGraphNode::visualize()
    {
      std::cout << "-----IterGraph Visualization------\n";
      applyAll();
      size_t n_tab = 0;
      for (auto common : commonIters)
      {
        for (size_t i = 0; i < n_tab; i++)
          std::cout << "\t";
        std::cout << "for  " << common->name << " in [0, " << common->ext << "):";
        if (common->shared)
          std::cout << "\"shared\" ";
        if (common->iv_type != IV_Type::REDUCE)
          std::cout << "\"parallel\"";
        std::cout << "\n";
        n_tab++;
      }
      size_t n_tab_ = n_tab;
      for (auto iv : firstOpIters)
      {
        if (iv->shared)
          continue;
        for (size_t i = 0; i < n_tab_; i++)
          std::cout << "\t";
        std::cout << "for  " << iv->name << " in [0, " << iv->ext << "):\n";
        n_tab_ += 1;
      }
      for (auto iv : firstOpTensorizeIters)
      {
        for (size_t i = 0; i < n_tab_; i++)
          std::cout << "\t";
        std::cout << "for  " << iv->name << " in [0, " << iv->ext
                  << ") \"tensorize\":\n";
        n_tab_ += 1;
      }
      n_tab_ = n_tab;
      for (size_t _ = attachPos; _ < secondOpIters.size(); _++)
      {
        auto iv = secondOpIters[_];
        for (size_t i = 0; i < n_tab_; i++)
          std::cout << "\t";
        std::cout << "for  " << iv->name << " in [0, " << iv->ext << "):\n";
        n_tab_++;
      }
      for (auto iv : secondOpTensorizeIters)
      {
        for (size_t i = 0; i < n_tab_; i++)
          std::cout << "\t";
        std::cout << "for  " << iv->name << " in [0, " << iv->ext
                  << ") \"tensorize\":\n";
        n_tab_ += 1;
      }
      std::cout << "---------------------------------\n";
    }
    bool ivBindingAndValidate(Array<IterVar> firstOpIters,
                              Array<IterVar> secondOpIters,
                              Array<Share> sharedPairs)
    {
      // op2.R bind to op1.S
      auto isMapped = [&sharedPairs](Array<IterVar> ivs1, Array<IterVar> ivs2)
      {
        bool tmp[ivs2.size()];
        if (!(ivs1.size() == ivs2.size()))
          return false;
        for (size_t i = 0; i < ivs2.size(); i++)
          tmp[i] = 0;
        for (auto iv1 : ivs1)
        {
          bool matched = false;
          for (auto share : sharedPairs)
          {
            if (share->upper == iv1)
            {
              for (size_t i = 0; i < ivs2.size(); i++)
              {
                if (!tmp[i] && ivs2[i] == share->lower)
                {
                  tmp[i] = true;
                  matched = true;
                  break;
                }
              }
              break;
            }
          }
          if (!matched)
            return false;
        }
        return true;
      };
      Array<IterVar> firstOpR, firstOpS1, firstOpS2, secondOpR, secondOpS1,
          secondOpS2;
      for (auto iv : secondOpIters)
        if (iv->iv_type == IV_Type::REDUCE)
          secondOpR.push_back(iv);
        else if (iv->iv_type == IV_Type::FIRSTSPATIAL)
          secondOpS1.push_back(iv);
        else if (iv->iv_type == IV_Type::SECONDSPATIAL)
          secondOpS2.push_back(iv);
      for (auto iv : firstOpIters)
        if (iv->iv_type == IV_Type::REDUCE)
          firstOpR.push_back(iv);
        else if (iv->iv_type == IV_Type::FIRSTSPATIAL)
          firstOpS1.push_back(iv);
        else if (iv->iv_type == IV_Type::SECONDSPATIAL)
          firstOpS2.push_back(iv);

      if (isMapped(firstOpS2, secondOpR))
      {
        // nothing to do
      }
      else if (isMapped(firstOpS1, secondOpR))
      {
        // switch firstOpS1 with firstOpS2
        Array<IterVar> tmp = firstOpS1;
        firstOpS1 = firstOpS2;
        firstOpS2 = tmp;
        for (auto iv : firstOpS1)
          iv->iv_type = IV_Type::FIRSTSPATIAL;
        for (auto iv : firstOpS2)
          iv->iv_type = IV_Type::SECONDSPATIAL;
      }
      else
      {
        std::cout << "secondOpR: " << secondOpR << std::endl;
        std::cout << "firstOpS1: " << firstOpS1 << std::endl;
        std::cout << "firstOpS2: " << firstOpS2 << std::endl;
        CHECK(false) << "secondOpR cannot match firstOpS";
      }

      if (isMapped(firstOpS1, secondOpS1))
      {
        // nothing to do
      }
      else if (isMapped(firstOpS1, secondOpS2))
      {
        // switch secondOpS1 with secondOpS2
        Array<IterVar> tmp = secondOpS1;
        secondOpS1 = secondOpS2;
        secondOpS2 = tmp;
        for (auto iv : secondOpS1)
          iv->iv_type = IV_Type::FIRSTSPATIAL;
        for (auto iv : secondOpS2)
          iv->iv_type = IV_Type::SECONDSPATIAL;
      }
      else
      {
        CHECK(false) << "firstOpS1 cannot match secondOpS";
      }
      return true;
    }
    inline IterGraph buildIterGraph(SerialFusionState sfState)
    {
      Array<tir::IterVar> tensorizeAxes = sfState->tensorizeAxes;
      OpHyperState ops1, ops2;
      std::tie(ops1, ops2) = sfState->getCubicOpPair();
      Array<IterVar> firstOpIters_ = ops1->getAllIters();
      Array<IterVar> secondOpIters_ = ops2->getAllIters();
      Array<Array<tir::IterVar>> shared_axis =
          share_axis_analysis(ops1->op, ops2->op, tensorizeAxes);
      std::unordered_map<tir::IterVar, IterVar> IterMap = ops1->getIterMap();
      IterMap.insert(ops2->getIterMap().begin(), ops2->getIterMap().end());
      Array<Share> sharedIterPairs;
      for (auto sharePair : shared_axis)
      {
        CHECK(IterMap.count(sharePair[0])) << sharePair[0] << " not in the IterMap";
        CHECK(IterMap.count(sharePair[1])) << sharePair[1] << " not in the IterMap";
        sharedIterPairs.push_back(
            Share(IterMap.at(sharePair[0]), IterMap.at(sharePair[1])));
      }
      bool isBindingCorrect =
          ivBindingAndValidate(firstOpIters_, secondOpIters_, sharedIterPairs);
      CHECK(isBindingCorrect) << "iv binding doesn't follow mlkn pattern";
      Array<IterVar> firstOpIters, secondOpIters, firsrOpTensorizeIters,
          secondOpTensorizeAxis;
      for (auto iv : firstOpIters_)
      {
        if (!(iv->iv_type == IV_Type::TENSORIZE))
          firstOpIters.push_back(iv);
        else
          firsrOpTensorizeIters.push_back(iv);
      }
      for (auto iv : secondOpIters_)
      {
        if (!(iv->iv_type == IV_Type::TENSORIZE))
          secondOpIters.push_back(iv);
        else
          secondOpTensorizeAxis.push_back(iv);
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
                       readProducerPos, ops1->op, ops2->op, firsrOpTensorizeIters,
                       secondOpTensorizeAxis);
    }

    TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<IterGraphNode>([](const ObjectRef &node, ReprPrinter *p)
                                     {
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
      p->stream << ",\n";
      p->stream << "_firstOpTensorizeIters:\t";
      p->Print(op->firstOpTensorizeIters);
      p->stream << ",\n";
      p->stream << "_secondOpTensorizeIters:\t";
      p->Print(op->secondOpTensorizeIters);
      p->stream << "\n------------------------------------------"; });

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
    TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setConfig")
        .set_body_method<IterGraph>(&IterGraphNode::setConfig);
  } // namespace auto_tensorize

} // namespace ditto