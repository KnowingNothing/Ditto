#include "../build_for_ops.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "compute_op.h"
#include "../../../3rdparty/tvm/src/te/operation/compute_op.h"
#include "op_utils.h"
#include "../../../3rdparty/tvm/src/te/operation/op_utils.h"

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {
using namespace tir;

Stmt TensorComputeOpNodeBuildProvide(const HybridStage& stage,
                                       const std::unordered_map<IterVar, Range>& dom_map,
                                       bool debug_keep_trivial_loop){
  const TensorComputeOpNode* this_ = stage->op.as<TensorComputeOpNode>();
  ICHECK_EQ(stage->op.operator->(), this_);

  // Start bind data.
  Stmt nop = Evaluate(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = this_->InputTensors();

  // input binding
  size_t num_inputs = inputs.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor tensor = inputs[i];
    Region region = this_->input_regions[i];
    Buffer buffer = this_->intrin->buffers[i];
    Array<ObjectRef> bind_spec{buffer, tensor};

    Array<PrimExpr> tuple;
    for (size_t i = 0; i < region.size(); ++i) {
      tuple.push_back(region[i]->min);
      tuple.push_back(region[i]->extent);
    }
    input_bind_nest.emplace_back(
        AttrStmt(bind_spec, tir::attr::buffer_bind_scope,
                 Call(DataType::Handle(), tir::builtin::tvm_tuple(), tuple), nop));
  }

  // output binding
  for (int i = 0; i < this_->num_outputs(); ++i) {
    Tensor tensor = stage->op.output(i);
    Buffer buffer = this_->intrin->buffers[num_inputs + i];
    Array<ObjectRef> bind_spec{buffer, tensor};

    Array<PrimExpr> tuple;
    for (size_t i = 0; i < this_->axis.size(); ++i) {
      auto ivar = this_->axis[i];
      if (i < static_cast<size_t>(this_->schedulable_ndim)) {
        tuple.push_back(ivar->var);
        tuple.push_back(1);
      } else {
        Range dom = ivar->dom;
        tuple.push_back(dom->min);
        tuple.push_back(dom->extent);
      }
    }

    output_bind_nest.emplace_back(
        AttrStmt(bind_spec, tir::attr::buffer_bind_scope,
                 Call(DataType::Handle(), tir::builtin::tvm_tuple(), tuple), nop));
  }

  // Check variable remap
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  tir::ArgBinder binder(&vmap);

  // Map the expressions passed in the call to the TensorIntrin, to the placeholder
  // variables
  Array<PrimExpr> user_expr = this_->scalar_inputs;
  Array<Var> scalar_params = this_->intrin->scalar_params;
  Array<PrimExpr> sp_expr;
  for (auto sp : scalar_params) {
    PrimExpr esp = sp;
    sp_expr.push_back(esp);
  }
  ICHECK_EQ(sp_expr.size(), user_expr.size());
  // TODO(jdavies-huawei): what name should be used here?
  binder.BindArray(sp_expr, user_expr, this_->name);

  size_t tloc = stage->leaf_iter_vars.size();
  ComputeLoopNest n = ComputeLoopNest::Create(this_, stage, dom_map, debug_keep_trivial_loop);

  if (this_->reduce_axis.size() == 0) {
    std::vector<std::vector<Stmt> > nest(n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(MakeIfNest(n.main_predicates));
    ICHECK_EQ(n.init_predicates.size(), 0U);
    ICHECK(this_->intrin->body.defined())
        << "Normal store op for intrin " << this_ << " is not defined";
    Stmt body = MergeNest(output_bind_nest, this_->intrin->body);
    body = MergeNest(input_bind_nest, body);
    body = tir::Substitute(body, vmap);
    body = MergeNest(binder.asserts(), body);
    body = te::Substitute(body, n.main_vmap);
    Stmt ret = MergeNest(nest, body);
    return ret;
  } else {
    // Need to split reduction
    ICHECK(this_->intrin->reduce_update.defined()) << "Reduction update op is not defined";
    // Need init and update steps
    ICHECK_NE(this_->reduce_axis.size(), 0U);
    std::vector<std::vector<Stmt> > common(n.main_nest.begin(),
                                           n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > update_nest(n.main_nest.begin() + n.num_common_loop + 1,
                                                n.main_nest.begin() + tloc + 1);
    update_nest.emplace_back(MakeIfNest(n.main_predicates));

    if (this_->intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(n.init_nest.begin(),
                                                n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(MakeIfNest(n.init_predicates));
      Stmt init = MergeNest(output_bind_nest, this_->intrin->reduce_init);
      init = te::Substitute(init, n.init_vmap);
      init = MergeNest(init_nest, init);
      // The update
      Stmt update = MergeNest(output_bind_nest, this_->intrin->reduce_update);
      update = MergeNest(input_bind_nest, update);
      update = tir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = te::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, SeqStmt::Flatten(init, update));
    } else {
      // When init op is not available, use body op for reset in the first iter.
      ICHECK(this_->intrin->body.defined()) << "Normal body op is not defined";
      Stmt update =
          TransformUpdate(stage, dom_map, n, this_->intrin->body, this_->intrin->reduce_update);
      update = MergeNest(output_bind_nest, update);
      update = MergeNest(input_bind_nest, update);
      update = tir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = te::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, update);
    }
  }
}

}  // namespace hybrid
}  // namespace ditto
