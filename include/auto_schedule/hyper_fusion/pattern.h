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

bool IsCubic(te::Operation op, int substantial);

bool IsAllred(te::Operation op, int substantial);

bool IsShuffle(te::Operation op);

bool IsLocal(te::Operation op, int substantial);

bool IsView(te::Operation op);

} // namespace auto_schedule

} // namespace ditto