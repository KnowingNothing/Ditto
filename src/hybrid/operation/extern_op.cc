#include "../build_for_ops.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_set>

#include "op_utils.h"
#include "../../../3rdparty/tvm/src/te/operation/op_utils.h"

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {
using namespace tir;

Stmt ExternOpNodeBuildRealize(const HybridStage& stage,
                                const std::unordered_map<IterVar, Range>& realize_map,
                                const Stmt& body, String storage_scope){
  const ExternOpNode* this_ = stage->op.as<ExternOpNode>();
  ICHECK_EQ(stage->op.get(), this_);
  Stmt realize_body = body;
  for (int k = 0; k < this_->num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      bounds.push_back(Range::FromMinExtent(make_const(t->shape[i].dtype(), 0), t->shape[i]));
    }
    realize_body = tir::ProducerRealize(t, bounds, const_true(), realize_body, storage_scope);
  }
  return realize_body;
}

Stmt ExternOpNodeBuildProvide(const HybridStage& stage,
                                const std::unordered_map<IterVar, Range>& dom_map,
                                bool debug_keep_trivial_loop){
  const ExternOpNode* this_ = stage->op.as<ExternOpNode>();
  ICHECK_EQ(stage->op.operator->(), this_);
  Stmt ret = AttrStmt(make_zero(DataType::Int(32)), tir::attr::extern_scope, 0, this_->body);
  auto f_push_bind = [&ret](Buffer buffer, Tensor tensor) {
    Array<ObjectRef> bind_spec;
    Array<PrimExpr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].dtype(), 0));
      tuple.push_back(buffer->shape[k]);
    }
    ret = AttrStmt(bind_spec, tir::attr::buffer_bind_scope,
                   Call(DataType::Handle(), builtin::tvm_tuple(), tuple), ret);
  };
  for (size_t i = this_->output_placeholders.size(); i != 0; --i) {
    f_push_bind(this_->output_placeholders[i - 1], stage->op.output(i - 1));
  }
  for (size_t i = this_->inputs.size(); i != 0; --i) {
    f_push_bind(this_->input_placeholders[i - 1], this_->inputs[i - 1]);
  }
  return ret;
}

}  // namespace hybrid
}  // namespace ditto
