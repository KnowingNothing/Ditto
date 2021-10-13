#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <hybrid/hybrid_schedule_pass.h>
#include <tvm/tir/expr_functor.h>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

using namespace tir;

class ElemWiseDetector : public tir::ExprVisitor {
 public:
  explicit ElemWiseDetector(Array<IterVar> axis) : axis_(axis) {}

  void VisitExpr(const PrimExpr& e) final {
    if (!is_elem_wise_) return;
    ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const ProducerLoadNode* op) final {
    Array<PrimExpr> indices = op->indices;
    if (axis_.size() != indices.size()) {
      is_elem_wise_ = false;
      return;
    }

    for (size_t i = 0; i < axis_.size(); ++i) {
      if (!indices[i].same_as(axis_[i]->var)) {
        is_elem_wise_ = false;
        return;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  bool is_elem_wise_{true};

 private:
  Array<IterVar> axis_;
};

bool IsElemWise(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    ElemWiseDetector v = ElemWiseDetector(compute->axis);
    for (auto& e : compute->body) v(e);
    return v.is_elem_wise_;
  }
  return false;
}

void AutoInlineElemWise(HybridSchedule sch) {
  for (HybridStage s : sch->stages) {
    if (!s.is_scheduled() && IsElemWise(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsBroadcast(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    if (compute->reduce_axis.size()) {
      return false;
    }
    constexpr auto kBroadcast = "broadcast";
    // broadcast op in topi has tag `broadcast`
    if (op->tag == kBroadcast) {
      return true;
    }
  }
  return false;
}

void AutoInlineBroadcast(HybridSchedule sch) {
  for (HybridStage s : sch->stages) {
    if (!s.is_scheduled() && IsBroadcast(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsInjective(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    return compute->reduce_axis.size() == 0;
  }
  return false;
}

void AutoInlineInjective(HybridSchedule sch) {
  for (HybridStage s : sch->stages) {
    if (!s.is_scheduled() && IsInjective(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

TVM_REGISTER_GLOBAL("ditto.AutoInlineElemWise").set_body_typed(AutoInlineElemWise);

TVM_REGISTER_GLOBAL("ditto.AutoInlineBroadcast").set_body_typed(AutoInlineBroadcast);

TVM_REGISTER_GLOBAL("ditto.AutoInlineInjective").set_body_typed(AutoInlineInjective);

}  // namespace hybrid
}  // namespace ditto
