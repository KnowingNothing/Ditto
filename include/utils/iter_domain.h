#pragma once

#include <tvm/arith/int_set.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_map>

namespace ditto {
using namespace tvm;
namespace utils {

std::unordered_map<const tir::VarNode *, tir::IterVarType> InferIterVarType(
    const Map<tir::Var, PrimExpr> &vars_to_infer,
    const std::unordered_map<const tir::VarNode *, tir::IterVarType>
        &ori_types);

Map<tir::Var, Range> InferRange(const Map<tir::Var, PrimExpr> &vars_to_infer,
                                const Map<tir::Var, Range> &ori_ranges);

Array<Array<PrimExpr>> GetAccessIndices(te::Operation op,
                                        te::Operation producer);
Array<tir::Var> GetAccessVars(te::Operation op, te::Operation producer);
PrimExpr ReplaceVars(PrimExpr expr, Map<tir::Var, tir::Var> map);
PrimExpr ReplaceVars(PrimExpr expr, Map<tir::Var, PrimExpr> map);

float GetFloatOps(PrimExpr body);

} // namespace utils

} // namespace ditto