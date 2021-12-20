#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
namespace ditto {
namespace auto_schedule {

class VarIn : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  VarIn(tir::Var var) : var_(var) {}

  bool check_exist(const PrimExpr &expr) {
    counter_ = 0;
    VisitExpr(expr);
    return counter_ > 0;
  }

  int count(const PrimExpr &expr) {
    counter_ = 0;
    VisitExpr(expr);
    return counter_;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::VarNode *op) override {
    if (op == var_.get()) {
      counter_ += 1;
    }
  }

private:
  tir::Var var_;
  int counter_{0};
};

class IndexIn : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  IndexIn(te::Operation op, tir::Var var) : op_(op), var_(var) {}

  bool check_exist(const PrimExpr &expr) {
    counter_ = 0;
    VisitExpr(expr);
    return counter_ > 0;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::ProducerLoadNode *op) override {
    te::Tensor t = runtime::Downcast<te::Tensor>(op->producer);
    VarIn vi(var_);
    if (t.defined() && t->op == op_) {
      for (auto idx : op->indices) {
        counter_ += vi.count(idx);
      }
    }
  }

private:
  te::Operation op_;
  tir::Var var_;
  int counter_{0};
};

PrimExpr flatten_indices(const Array<PrimExpr> shape,
                         const Array<PrimExpr> indices) {
  int num_dim = (int)shape.size();
  CHECK((int)indices.size() == num_dim && num_dim > 0);
  PrimExpr flatten = indices[num_dim - 1];
  for (int i = 0; i < num_dim - 1; ++i) {
    flatten = indices[num_dim - i - 2] * shape[num_dim - i - 1] + flatten;
  }
  return flatten;
}

class FlattenIndices : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  FlattenIndices(te::Operation op) : op_(op) {}

  Array<PrimExpr> flatten(const PrimExpr &expr) {
    flatten_.clear();
    VisitExpr(expr);
    return flatten_;
  }

protected:
  using tir::ExprVisitor::VisitExpr_;
  void VisitExpr_(const tir::ProducerLoadNode *op) override {
    te::Tensor t = runtime::Downcast<te::Tensor>(op->producer);
    if (t.defined() && t->op == op_) {
      PrimExpr tmp = flatten_indices(t->shape, op->indices);
      flatten_.push_back(tmp);
    }
  }

private:
  te::Operation op_;
  Array<PrimExpr> flatten_;
};

bool IsCubic(te::Operation op, int substantial);

bool IsAllred(te::Operation op, int substantial);

bool IsShuffle(te::Operation op);

bool IsLocal(te::Operation op, int substantial);

bool IsView(te::Operation op);

} // namespace auto_schedule

} // namespace ditto