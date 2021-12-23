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
namespace auto_tensorize {

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

class IndexInDim : public tir::ExprVisitor {
public:
  using tir::ExprVisitor::VisitExpr;
  IndexInDim(te::Operation op, tir::Var var, int dim)
      : op_(op), var_(var), dim_(dim) {}

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
    if (t.defined() && t->op == op_ && dim_ < (int)op->indices.size()) {
      counter_ += vi.count(op->indices[dim_]);
    }
  }

private:
  te::Operation op_;
  tir::Var var_;
  int dim_{0};
  int counter_{0};
};

PrimExpr flatten_indices(const Array<PrimExpr> shape,
                         const Array<PrimExpr> indices);

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

Array<Array<tir::IterVar>> share_axis_analysis(te::Operation op1,
                                               te::Operation op2);

} // namespace auto_tensorize

} // namespace ditto