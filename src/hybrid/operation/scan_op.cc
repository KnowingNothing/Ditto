#include "../build_for_ops.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include "../graph.h"
#include "../../../3rdparty/tvm/src/te/schedule/graph.h"
#include "op_utils.h"
#include "../../../3rdparty/tvm/src/te/operation/op_utils.h"

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {
using namespace tir;

Stmt ScanOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                              const Stmt& body, String storage_scope){
  const ScanOpNode* this_ = stage->op.as<ScanOpNode>();
  arith::Analyzer analyzer;
  ICHECK_EQ(stage->op.get(), this_);
  Range sdom = dom_map.at(this_->scan_axis);
  Range tdom = Range::FromMinExtent(0, analyzer.Simplify(sdom->extent + sdom->min));
  Stmt ret = body;
  size_t sp_idx = 0;
  for (size_t i = 0; i < this_->update.size(); ++i) {
    Tensor t = stage->op.output(i);
    ICHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    bounds.push_back(tdom);
    for (size_t k = 1; k < this_->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this_->spatial_axis_[sp_idx];
      bounds.push_back(dom_map.at(sp_ax));
    }
    ret = tir::ProducerRealize(t, bounds, const_true(), ret, storage_scope);
  }
  return ret;
}

Stmt ScanOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                              bool debug_keep_trivial_loop){
  const ScanOpNode* this_ = stage->op.as<ScanOpNode>();
  ICHECK_EQ(stage->op.operator->(), this_);
  Stmt provide =
      AttrStmt(stage->op, tir::attr::scan_update_scope, this_->scan_axis->var, Evaluate(0));
  Stmt init = AttrStmt(stage->op, tir::attr::scan_init_scope, 0, Evaluate(0));
  size_t begin_scan = 0;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    if (stage->leaf_iter_vars[i]->iter_type == kThreadIndex) {
      ICHECK_EQ(begin_scan, i);
      begin_scan = i + 1;
    }
  }
  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest[begin_scan].push_back(init);
  nest.push_back(MakeIfNest(MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  return MergeNest(nest, provide);
}

}  // namespace hybrid
}  // namespace ditto
