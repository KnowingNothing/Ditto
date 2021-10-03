#include <tvm/tir/expr_functor.h>
#include <utils/iter_domain.h>

namespace ditto {
using namespace tvm;
namespace utils {

class IterTypeInfer : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  IterTypeInfer(
      std::unordered_map<const tir::VarNode *, tir::IterVarType> var_types)
      : var_types_(var_types) {}

  tir::IterVarType infer(const PrimExpr &expr) {
    type_ = tir::IterVarType::kTensorized;
    VisitExpr(expr);
    return type_;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::VarNode *op) override {
    CHECK(var_types_.count(op)) << "Can't find var " << op << ".\n";
    if (type_ == tir::IterVarType::kTensorized) {
      type_ = var_types_.at(op);
    } else {
      CHECK(type_ == var_types_.at(op))
          << "Can't blend type " << type_ << " with type " << var_types_.at(op)
          << ".\n";
    }
  }

private:
  std::unordered_map<const tir::VarNode *, tir::IterVarType> var_types_;
  tir::IterVarType type_{
      tir::IterVarType::kTensorized}; // use an impossible type
};

std::unordered_map<const tir::VarNode *, tir::IterVarType> InferIterVarType(
    const Map<tir::Var, PrimExpr> &vars_to_infer,
    const std::unordered_map<const tir::VarNode *, tir::IterVarType>
        &ori_types) {
  // The resulting types
  std::unordered_map<const tir::VarNode *, tir::IterVarType> new_types;

  IterTypeInfer infer(ori_types);
  for (auto kv : vars_to_infer) {
    tir::IterVarType type = infer.infer(kv.second);
    new_types[kv.first.get()] = type;
  }
  return new_types;
}

Map<tir::Var, Range> InferRange(const Map<tir::Var, PrimExpr> &vars_to_infer,
                                const Map<tir::Var, Range> &ori_ranges) {
  // The resulting ranges
  Map<tir::Var, Range> new_ranges;

  std::unordered_map<const tir::VarNode *, arith::IntSet> var_intsets;
  for (const auto &p : ori_ranges) {
    // Convert original ranges to IntSets
    var_intsets[p.first.get()] = arith::IntSet::FromRange(p.second);
  }

  // Infer ranges for the new variables and add them to the resulting ranges
  for (const auto &p : vars_to_infer) {
    const auto &var = p.first;
    const auto &expr = p.second;
    Range range = arith::EvalSet(expr, var_intsets).CoverRange(Range());
    if (range.defined()) {
      new_ranges.Set(var, range);
    }
  }
  return new_ranges;
}

} // namespace utils

} // namespace ditto