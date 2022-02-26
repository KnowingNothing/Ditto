#include <tvm/runtime/registry.h>
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

class IndicesGetter : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;

  Array<Array<PrimExpr>> get(const PrimExpr &expr, const te::Operation &op) {
    indices_.clear();
    op_ = op;
    VisitExpr(expr);
    return Array<Array<PrimExpr>>(indices_);
  }

protected:
  using tir::ExprVisitor::VisitExpr_;

  void VisitExpr_(const tir::ProducerLoadNode *op) override {
    te::Tensor t = runtime::Downcast<te::Tensor>(op->producer);
    if (t.defined() && t->op == op_) {
      indices_.push_back(op->indices);
    }
  }

private:
  std::vector<Array<PrimExpr>> indices_;
  te::Operation op_{nullptr};
};

Array<Array<PrimExpr>> GetAccessIndices(te::Operation op,
                                        te::Operation producer) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop && cop->body.size() == 1U);
  IndicesGetter getter;
  return getter.get(cop->body[0], producer);
}

class VarsGetter : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;

  Array<tir::Var> get(const PrimExpr &expr, const te::Operation &op) {
    vars_.clear();
    op_ = op;
    VisitExpr(expr);
    return vars_;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;

  void VisitExpr_(const tir::ProducerLoadNode *op) override {
    te::Tensor t = runtime::Downcast<te::Tensor>(op->producer);
    if (t.defined() && t->op == op_) {
      for (auto idx : op->indices)
        this->VisitExpr(idx);
    }
  }
  void VisitExpr_(const tir::VarNode * op) override{
    this->vars_.push_back(GetRef<tir::Var>(op));
  }

private:
  Array<tir::Var> vars_;
  te::Operation op_{nullptr};
};

Array<tir::Var> GetAccessVars(te::Operation op, te::Operation producer){
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  CHECK(cop && cop->body.size() == 1U);
  VarsGetter getter;
  return getter.get(cop->body[0], producer);
}
class FloatOpGetter : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;

  FloatOpGetter() : count_(0) {}

  int Get(const PrimExpr &expr) {
    count_ = 0;
    VisitExpr(expr);
    return count_;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::AddNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::SubNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::MulNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::DivNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::ModNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::FloorDivNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::FloorModNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::MinNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::MaxNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::EQNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::NENode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::LTNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::LENode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::GTNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::GENode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::AndNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::OrNode *op) override {
    if (op->a->dtype.is_float() || op->b->dtype.is_float()) {
      count_ += 1;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const tir::ReduceNode *op) override {
    tir::ExprVisitor::VisitExpr_(op);
    int mul = 1;
    for (auto iv : op->axis) {
      const IntImmNode *as_int = iv->dom->extent.as<IntImmNode>();
      CHECK(as_int) << "Please provide constant bound, instead of "
                    << iv->dom->extent << ".\n";
      mul = mul * as_int->value;
    }
    count_ = count_ * mul;
  }

private:
  int count_;
};

float GetFloatOps(PrimExpr body) {
  FloatOpGetter getter;
  return (float)getter.Get(body);
}

TVM_REGISTER_GLOBAL("ditto.utils.InferRange").set_body_typed(InferRange);
TVM_REGISTER_GLOBAL("ditto.utils.GetAccessIndices")
    .set_body_typed(GetAccessIndices);

} // namespace utils

} // namespace ditto