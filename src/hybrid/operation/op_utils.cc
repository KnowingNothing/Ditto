#include "op_utils.h"

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <string>

#include "../../../3rdparty/tvm/src/runtime/thread_storage_scope.h"
#include "../message_passing.h"

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

using namespace arith;
using namespace tir;

template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

std::vector<std::vector<Stmt> > MakeLoopNest(const HybridStage& stage,
                                             const std::unordered_map<IterVar, Range>& dom_map,
                                             size_t begin_iter_pos, bool new_loop_var,
                                             const std::unordered_set<IterVar>& skip_iter,
                                             std::unordered_map<IterVar, PrimExpr>* p_value_map,
                                             bool debug_keep_trivial_loop, 
                                            std::unordered_map<IterVar, int> update_state) {
  auto leaf_iter_vars = stage->leaf_iter_vars;
  const Tree<IterVar> & leaf_iter_vars_tree = stage->leaf_iter_vars_tree;
  Stmt no_op = Evaluate(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  // handle pinpt for serial
  std::unordered_map<IterVar, int> serial_pinpt_map;
  for (IterVarRelation rel : stage->relations) {
    if (const SliceNode* r = rel.as<SliceNode>()){
      if(r->mode == "serial"){
        serial_pinpt_map[r->pinpt] = 1;
      }
    }
  }
  PassDownBitMaskOr_WithoutSlice(stage, &serial_pinpt_map, true);

  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    // Bind iv could be another thread.
    IterVar bind_iv = iv;
    if (stage->iter_var_attrs.count(iv)) {
      IterVar bind_thread = stage->iter_var_attrs[iv]->bind_thread;
      if (bind_thread.defined()) bind_iv = bind_thread;
    }

    Range dom = dom_map.at(iv);

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForKind kind = ForKind::kSerial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled:
            kind = ForKind::kUnrolled;
            break;
          case kVectorized:
            kind = ForKind::kVectorized;
            break;
          case kParallelized:
            kind = ForKind::kParallel;
            break;
          case kDataPar:
            break;
          case kTensorized:
            break;
          default:
            LOG(FATAL) << "Unknown iter type" << it_attr->iter_type << " in the iter_var_attrs";
        }
        ICHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImmNode>()->value;
          PrimExpr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(DataType::Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmt(iv, tir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      bool is_under_pinpt = false;
      const SliceNode* rr = NULL;
      for (IterVarRelation rel : stage->relations) {
        if (const SliceNode* r = rel.as<SliceNode>()){
          if(r->left[0].same_as(iv) || r->right[0].same_as(iv)){
            is_under_pinpt = true;
            rr = r;
          }
        }
      }
      if(is_under_pinpt){
        value_map[iv] = var;
        // not insert here, because of reorder
        if(rr->left[0].same_as(iv)){
          nest[i+1].emplace_back(LetStmt(rr->sel, 1, no_op));
        }
        else{
          nest[i+1].emplace_back(LetStmt(rr->sel, 0, no_op));
        }
        // handle slice parallel
        if(rr->mode == "parallel"){
          nest[i+1].emplace_back(LetStmt(var, rr->pinpt->var, no_op));
          // don't know whether it is factor or factor+min
          if(rr->left[0].same_as(iv)){
            nest[i+1].emplace_back(IfThenElse(var<=rr->factor, no_op));
          }
          else{
            nest[i+1].emplace_back(IfThenElse(var>rr->factor, no_op));
          }
          continue;
        }
      }
      if(serial_pinpt_map[iv]){
        value_map[iv] = var;
        continue;
      }
      if(new_loop_var&& (update_state.at(iv) & 2)){
        // do nothing
      }
      else if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(LetStmt(var, cast(var.dtype(), dom->min), no_op));
        value_map[iv] = cast(var.dtype(), dom->min);
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(For(var, 0, dom->extent, kind, no_op));
        value_map[iv] = var;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.dtype());
        nest[i + 1].emplace_back(For(idx, 0, dom->extent, kind, no_op));
        PrimExpr new_value = dom->min + idx;
        // here: modify original stmt
        // put the analysis of var=new_value later
        // inorder to support let i=0 when meeting a branch
        // value_map[iv] = new_value;
        value_map[iv] = var;
        nest[i + 1].emplace_back(LetStmt(var, new_value, no_op));
      }
      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        ICHECK(!is_one(dom->extent)) << "Cannot prefetch on trivial loop with extent=1";
        ICHECK_EQ(it_attr->prefetch_data.size(), it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(AttrStmt(it_attr->prefetch_data[j], tir::attr::prefetch_scope,
                                            it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" || bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      ICHECK(is_zero(dom->min));
      ICHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(AttrStmt(bind_iv, tir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      ICHECK(is_zero(dom->min));
      ICHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt(bind_iv, tir::attr::pipeline_exec_scope, dom->extent, no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      ICHECK(is_zero(dom->min)) << "Itervar " << iv << " must start at zero, but it starts at "
                                << dom->min;
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(AttrStmt(bind_iv, tir::attr::thread_extent, dom->extent, no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else if (stage->scope == "") {
        value_map[iv] = var;
      } else {
        runtime::ThreadScope ts = runtime::ThreadScope::Create(bind_iv->thread_tag);
        runtime::StorageScope ss = runtime::StorageScope::Create(stage->scope);
        if (static_cast<int>(ss.rank) <= ts.rank) {
          value_map[iv] = var;
        } else if (stage->scope == "warp" && ts.rank == 1) {
          // To determine whether a thread index is inside or outside a warp, we need
          // to know the thread extent. We leave a warning for now.
          if (ts.dim_index == 0) {
            value_map[iv] = var;
          } else {
            LOG(WARNING)
                << "WARNING: threadIdx.y or threadIdx.z accessing warp-scope memory detected. "
                << "TVM assumes only threadIdx.x indicates threads inside a warp, "
                << "while threadIdx.y and threadIdx.z indicates different warps.";
            value_map[iv] = dom->min;
          }
        } else {
          value_map[iv] = dom->min;
        }
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(AttrStmt(iv, tir::attr::loop_scope, iv->var, no_op));
    }
  }
  /*
  bool flag = false;
  std::unordered_map<IterVar, const SliceNode *> rmap; // reverse map from pinpt itervar to slicenode
  for (const IterVarRelation rel : stage->relations){
    if(const SliceNode * s = rel.as<SliceNode>()){
      update_state[s->pinpt] |= 4; // mark the pinpt
      rmap[s->pinpt] = s;
      ICHECK(!flag) << "should be no slice after rebase";
    }
    else if (const RebaseNode *s = rel.as<RebaseNode>()){
      update_state[s->rebased] |= update_state[s->parent]; // pass to rebase node;
      if (update_state[s->rebased] & 4){
        if (rmap.count(s->parent))
          rmap[s->rebased] = rmap.at(s->parent);
        else 
          std::cout << s->parent << " has no rmap" << std::endl;
      } 
      flag = true;
    }
  }
  std::stack<TreeUnitNode<IterVar> *> S;
  S.push(leaf_iter_vars_tree.getBase());
  while(!S.empty()){
    TreeUnitNode<IterVar> * tmp = S.top();
    S.pop();
    if(!tmp) continue;
    if(tmp->data_ptr){
      if (update_state.at(*(tmp->data_ptr)) & 4){ // is pinpt
        TreeUnitNode<IterVar> * child = tmp->pChild;
        int sel = 1;
        while(child){
          size_t pos = FindNodeRef(stage->leaf_iter_vars.GetArrayNode(), *child->data_ptr) + 1;
          ICHECK(rmap.count(*(tmp->data_ptr))) << *(tmp -> data_ptr) << "failed";
          nest[pos].insert(nest[pos].begin(), LetStmt(rmap.at(*(tmp->data_ptr))->sel, sel --, no_op));
          child = child->pSibling;
        }
        ICHECK(sel == -1) << "pinpt should not have more than 2 child";
      }
    }
    TreeUnitNode<IterVar> * child = tmp->pChild;
    while(child){
      S.push(child);
      child = child->pSibling;
    }
  }*/
  // message passing to get offset of root iter vars.
  hybrid::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

}  // namespace hybrid
}  // namespace ditto
