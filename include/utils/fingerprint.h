#pragma once

#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <utils/logging.h>

namespace ditto {
using namespace tvm;
using namespace tvm::te;
using namespace tvm::tir;
namespace utils {

class ExprReMapper : public tir::ExprMutator {
public:
  using tir::ExprMutator::VisitExpr;
  PrimExpr VisitExpr_(const VarNode *op) final;

  PrimExpr VisitExpr_(const SizeVarNode *op) final;

  PrimExpr VisitExpr_(const ProducerLoadNode *op) final;

  PrimExpr VisitExpr_(const ReduceNode *op) final;

  ExprReMapper(Array<IterVar> axis) : count_var(0), count_call(0) {
    for (auto iv : axis) {
      var_map[iv->var.get()] = Var(get_new_var_name());
    }
  }

  Var get(const VarNode *k) {
    CHECK(var_map.count(k)) << "Unknown var " << GetRef<PrimExpr>(k) << ".\n";
    return var_map.at(k);
  }

private:
  std::string get_new_var_name() {
    int current = count_var++;
    return "v" + std::to_string(current);
  }

  std::string get_new_tensor_name() {
    int current = count_call++;
    return "T" + std::to_string(current);
  }

  te::Tensor get_new_tensor(const te::Tensor &t) {
    return te::placeholder(t->shape, t->dtype,
                           get_new_tensor_name() /*, t->requires_grad*/);
  }

  std::unordered_map<const VarNode *, Var> var_map;
  std::unordered_map<const SizeVarNode *, SizeVar> size_var_map;
  std::unordered_map<DataProducer, te::Tensor, ObjectHash, ObjectEqual>
      call_map;
  int count_var;
  int count_call;
};

std::string GetFingerPrint(Array<IterVar> axis, Array<PrimExpr> body);

} // namespace utils
} // namespace ditto