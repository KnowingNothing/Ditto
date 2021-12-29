#include <hardware/visa/scalar_isa.h>

namespace ditto {

namespace hardware {

ScalarISA::ScalarISA(String name, double latency, te::Operation func) {
  auto node = make_object<ScalarISANode>();
  node->name = name;
  node->latency = latency;
  node->func = func;
  data_ = node;
}

ScalarISA ScalarBinaryAdd(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr add =
      (te::Cast(res_dtype, lhs[iv->var]) + te::Cast(res_dtype, rhs[iv->var]));

  te::Operation func =
      te::ComputeOp("func", "scalar.binary_add", {}, {iv}, {add});
  return ScalarISA(name, latency, func);
}

ScalarISA ScalarBinarySub(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr sub =
      (te::Cast(res_dtype, lhs[iv->var]) - te::Cast(res_dtype, rhs[iv->var]));

  te::Operation func =
      te::ComputeOp("func", "scalar.binary_sub", {}, {iv}, {sub});
  return ScalarISA(name, latency, func);
}

ScalarISA ScalarBinaryMul(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr mul =
      (te::Cast(res_dtype, lhs[iv->var]) * te::Cast(res_dtype, rhs[iv->var]));

  te::Operation func =
      te::ComputeOp("func", "scalar.binary_mul", {}, {iv}, {mul});
  return ScalarISA(name, latency, func);
}

ScalarISA ScalarBinaryDiv(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr div = te::Div(te::Cast(res_dtype, lhs[iv->var]),
                         te::Cast(res_dtype, rhs[iv->var]));

  te::Operation func =
      te::ComputeOp("func", "scalar.binary_div", {}, {iv}, {div});
  return ScalarISA(name, latency, func);
}

ScalarISA ScalarBinaryMod(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr mod = te::Mod(te::Cast(res_dtype, lhs[iv->var]),
                         te::Cast(res_dtype, rhs[iv->var]));

  te::Operation func =
      te::ComputeOp("func", "scalar.binary_mod", {}, {iv}, {mod});
  return ScalarISA(name, latency, func);
}

ScalarISA ScalarMultiplyAdd(String name, double latency,
                            runtime::DataType lhs_dtype,
                            runtime::DataType rhs_dtype,
                            runtime::DataType res_dtype) {
  te::Tensor lhs = te::placeholder({1}, lhs_dtype, "lhs");
  te::Tensor rhs = te::placeholder({1}, rhs_dtype, "rhs");
  te::Tensor acc = te::placeholder({1}, res_dtype, "acc");
  tir::IterVar iv = SpatialAxis(1);
  PrimExpr multiply_add =
      (te::Cast(res_dtype, lhs[iv->var]) * te::Cast(res_dtype, rhs[iv->var]) +
       acc[iv->var]);

  te::Operation func =
      te::ComputeOp("func", "scalar.multiply_add", {}, {iv}, {multiply_add});
  return ScalarISA(name, latency, func);
}

} // namespace hardware

} // namespace ditto