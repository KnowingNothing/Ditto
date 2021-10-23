#include "../build_for_ops.h"

#include "compute_op.h"
#include "../../../3rdparty/tvm/src/te/operation/compute_op.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "../../../3rdparty/tvm/src/arith/interval_set.h"
#include "../message_passing.h"
#include "op_utils.h"
#include "../../../3rdparty/tvm/src/te/operation/op_utils.h"

using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

Stmt MergeNestTreeDfs(const std::vector<std::vector<Stmt>>& nest, Stmt body, const HybridStage& stage, TreeUnitNode<IterVar>* iv){
  Array<Stmt> sa;
  if(iv->pChild == NULL) 
    return MergeNest(nest[FindNodeRef(stage->leaf_iter_vars.GetArrayNode(), *iv->data_ptr) + 1], body);
  TreeUnitNode<IterVar> * tmp = iv->pChild;
  while(tmp != NULL){
    sa.push_back(MergeNestTreeDfs(nest, body, stage, tmp));
    tmp = tmp->pSibling;
  }
  return MergeNest(nest[FindNodeRef(stage->leaf_iter_vars.GetArrayNode(), *iv->data_ptr) + 1], SeqStmt::Flatten(sa));
}

Stmt MergeNestTree(const std::vector<std::vector<Stmt>>& nest, Stmt body, const HybridStage& stage) {
  body = MergeNest(nest[nest.size() - 1], body);
  return MergeNestTreeDfs(nest, body, stage, stage->leaf_iter_vars_tree.getRoot());
}

Stmt BaseComputeOpNodeBuildRealize(const HybridStage& stage,
                                     const std::unordered_map<IterVar, Range>& realize_map,
                                     const Stmt& body, String storage_scope) {
  //BaseComputeOpNode* this_ = Downcast<BaseComputeOp>(stage->op);
  const BaseComputeOpNode* this_ = stage->op.as<BaseComputeOpNode>();
  ICHECK_EQ(stage->op.get(), this_);
  Region bounds;
  for (IterVar iv : this_->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  Stmt realize = body;
  for (int i = this_->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i - 1);
    realize = tir::ProducerRealize(t, bounds, const_true(), realize, storage_scope);
    // alignment requirement, only useful for compute
    for (size_t i = 0; i < this_->num_schedulable_dims(); ++i) {
      auto it = stage->iter_var_attrs.find(this_->axis[i]);
      if (it != stage->iter_var_attrs.end()) {
        IterVarAttr attr = (*it).second;
        if (attr->dim_align_factor != 0) {
          Array<PrimExpr> tuple = {static_cast<int>(i), attr->dim_align_factor,
                                   attr->dim_align_offset};
          realize =
              tir::AttrStmt(t, tir::attr::buffer_dim_align,
                            Call(DataType::Handle(), tir::builtin::tvm_tuple(), tuple), realize);
        }
      }
    }
  }
  return realize;
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op, const Array<Tensor>& tensors, Stmt* init,
                   Stmt* provide) {
  Array<PrimExpr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  std::vector<Stmt> inits, provides;

  size_t size = op->body.size();
  const ReduceNode* reduce = op->body[0].as<ReduceNode>();
  ICHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  ICHECK(combiner);
  Array<PrimExpr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(tensors[i](args));
  }
  Array<PrimExpr> init_value = combiner->identity_element;
  Array<PrimExpr> update_value = (*combiner)(lhs, reduce->source);

  // If an init was passed to ReduceNode, use that for initialization
  // instead of combiner->identity_element
  Array<PrimExpr> reduce_init = reduce->init;
  if (!reduce_init.empty()) {
    init_value = reduce_init;
  }
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(ProducerStore(t, init_value[i], args));
    provides.emplace_back(ProducerStore(t, update_value[i], args));
  }
  *init = SeqStmt::Flatten(inits);
  *provide = SeqStmt::Flatten(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElse(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const ComputeOpNode* op, const Tensor& t) {
  Array<PrimExpr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return ProducerStore(t, op->body[t->value_index], args);
}

Stmt MakeComputeStmt(const ComputeOpNode* self, const HybridStage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     bool debug_keep_trivial_loop) {
  // grab the nest structure
  ComputeLoopNest n = ComputeLoopNest::Create(self, stage, dom_map, debug_keep_trivial_loop);
  // Normal loop structure
  n.init_nest.emplace_back(MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(MakeIfNest(n.main_predicates));
  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      source.push_back(stage->op.output(i));
    }
    MakeReduction(self, source, &init, &provide);
    init = MergeNest(n.init_nest, init);
    init = Substitute(init, n.init_vmap);
    // common nest
    std::vector<std::vector<Stmt> > common(n.main_nest.begin(),
                                           n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > reduce(n.main_nest.begin() + n.num_common_loop + 1,
                                           n.main_nest.end());
    provide = MergeNest(reduce, provide);
    if (debug_keep_trivial_loop) {
      provide = MergeNest(common, provide);
    } else {
      provide = MergeNest(common, SeqStmt::Flatten(init, provide));
    }
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return Substitute(provide, n.main_vmap);
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(self, stage->op.output(i)));
    }
    Stmt provide = SeqStmt::Flatten(provides);
    // provide = MergeNest(n.main_nest, provide);
    // modify it to tree structure
    provide = MergeNestTree(n.main_nest, provide, stage);
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return Substitute(provide, n.main_vmap);
  }
}

enum class ComputeType { kNormal, kCrossThreadReduction, kTensorize };

ComputeType DetectComputeType(const ComputeOpNode* self, const HybridStage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0, tensorize = 0;

  for (IterVar iv : stage->leaf_iter_vars) {
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (attr.defined() && attr->iter_type == kTensorized) {
      ++tensorize;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      ICHECK_EQ(thread_red, 0) << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  if (tensorize != 0) {
    ICHECK(thread_red == 0) << "Cannot mix cross thread reduction with Tensorize";
    return ComputeType::kTensorize;
  }
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt ComputeOpNodeBuildProvide(const HybridStage& stage,
                                 const std::unordered_map<IterVar, Range>& dom_map,
                                 bool debug_keep_trivial_loop){
  const ComputeOpNode* this_ = stage->op.as<ComputeOpNode>();
  ICHECK_EQ(stage->op.operator->(), this_);
  ComputeType ctype = DetectComputeType(this_, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this_, stage, dom_map, debug_keep_trivial_loop);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this_, stage, dom_map, debug_keep_trivial_loop);
  } else {
    return MakeComputeStmt(this_, stage, dom_map, debug_keep_trivial_loop);
  }
}

ComputeLoopNest ComputeLoopNest::Create(const BaseComputeOpNode* self, const HybridStage& stage,
                                        const std::unordered_map<IterVar, Range>& dom_map,
                                        bool debug_keep_trivial_loop) {
  ICHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest
  ret.main_nest = MakeLoopNest(stage, dom_map, 0, false, std::unordered_set<IterVar>(),
                               &ret.main_vmap, debug_keep_trivial_loop);
  ret.main_predicates =
      MakeBoundCheck(stage, dom_map, ret.main_vmap, false, std::unordered_set<IterVar>());
  for (auto& e : ret.main_predicates) {
    e = likely(e);
  }
  if (stage->store_predicate.defined()) {
    ret.main_predicates.push_back(stage->store_predicate);
  }
  if (self->reduce_axis.size() != 0) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : self->reduce_axis) {
      update_state[iv] = 2;
    }
    for (size_t i = 0; i < self->num_schedulable_dims(); ++i) {
      update_state[self->axis[i]] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    hybrid::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i;
        break;
      }
      ret.init_vmap[iv] = ret.main_vmap.at(iv);
    }
    ret.num_common_loop = begin_loop;
    // skip loops that are related to reduction and are unrelated to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if (flag == 2) skip_iter.insert(kv.first);
    }
    ret.init_nest = MakeLoopNest(stage, dom_map, begin_loop, true, skip_iter, &(ret.init_vmap),
                                 debug_keep_trivial_loop);
    ret.init_predicates = MakeBoundCheck(stage, dom_map, ret.init_vmap, true, skip_iter);
    for (auto& e : ret.init_predicates) {
      e = likely(e);
    }
  } else {
    ICHECK_EQ(ret.main_nest.size(), stage->leaf_iter_vars.size() + 1);
    ret.num_common_loop = stage->leaf_iter_vars.size();
  }
  // copy elison here.
  return ret;
}

Stmt TransformUpdate(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n, Stmt body, Stmt update) {
  Array<PrimExpr> conds;
  std::unordered_set<const VarNode*> banned;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    IterVar iv = stage->leaf_iter_vars[i];
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (attr->iter_type == kTensorized) {
        break;
      }
    }
    if (iv->iter_type == kCommReduce) {
      auto vit = dom_map.find(iv);
      ICHECK(vit != dom_map.end());
      const Range& vrange = vit->second;
      conds.push_back(likely(iv->var > vrange->min));
      banned.insert(iv->var.get());
    }
  }

  auto fbanned = [&](const VarNode* node) { return banned.count(node); };

  for (const PrimExpr& pred : n.main_predicates) {
    if (tir::UsesVar(pred, fbanned)) {
      LOG(FATAL) << "Tensorize update transform failed, the condition " << pred
                 << " has a conflict with the reset condition";
    }
  }

  auto cond = foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_or(a, b, span); },
                    const_false(1), conds);
  return IfThenElse(cond, update, body);
}

} // namespace hybrid
} // namespace ditto
