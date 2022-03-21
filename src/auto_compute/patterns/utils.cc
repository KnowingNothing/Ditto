#include <auto_compute/patterns/utils.h>

namespace ditto {

namespace auto_compute {

void PureIndexIn::VisitExpr_(const tir::ProducerLoadNode *op) {
  te::Tensor t = runtime::Downcast<te::Tensor>(op->producer);
  if (t.defined() && t->op == op_) {
    for (auto idx : op->indices) {
      tir::ExprVisitor::VisitExpr(idx);
      const tir::VarNode *as_var = idx.as<tir::VarNode>();
      if (as_var != nullptr && as_var == var_.get()) {
        counter_ += 1;
      }
    }
  }
}

IntImm make_int(int value) { return IntImm(runtime::DataType::Int(32), value); }

} // namespace auto_compute

} // namespace ditto