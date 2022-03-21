#pragma once

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

namespace ditto {
using namespace tvm;
namespace auto_compute {

class PureIndexIn : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  PureIndexIn(te::Operation op, tir::Var var) : op_(op), var_(var) {}

  bool check_single(const PrimExpr &expr) {
    counter_ = 0;
    dim_ = 0;
    VisitExpr(expr);
    return counter_ == 1;
  }

  int get_dim() { return dim_; }

  bool check_exist(const PrimExpr &expr) {
    counter_ = 0;
    VisitExpr(expr);
    return counter_ > 0;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::ProducerLoadNode *op) override;

private:
  te::Operation op_;
  tir::Var var_;
  int counter_{0};
  int dim_{-1};
};

IntImm make_int(int value);

} // namespace auto_compute

} // namespace ditto