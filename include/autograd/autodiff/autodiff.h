#pragma once

#include <tvm/runtime/object.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

namespace ditto {
using namespace tvm;
using namespace tvm::te;
using namespace tvm::tir;
namespace autograd {

#define UNEXPECTED                                                             \
  {                                                                            \
    LOG(FATAL) << "Unexpected visit: " << GetRef<PrimExpr>(op);                \
    throw;                                                                     \
  }

TVM_DLL Array<Tensor> Gradient(const Tensor &output,
                               const Array<Tensor> &inputs,
                               const Tensor &head = Tensor());

TVM_DLL bool expr_equal(const PrimExpr &a, const PrimExpr &b);

TVM_DLL Tensor grad_op(const Tensor &input, const Tensor &output,
                       const Tensor &doutput);

} // namespace autograd
} // namespace ditto
