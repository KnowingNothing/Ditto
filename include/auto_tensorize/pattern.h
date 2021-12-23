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

#include <auto_tensorize/config.h>

using namespace tvm;
namespace ditto {
namespace auto_tensorize {

enum class OpPattern : int {
  PATTERN_CUBIC = 0,
  PATTERN_ALLRED = 1,
  PATTERN_SHUFFLE = 2,
  PATTERN_LOCAL = 3,
  PATTERN_VIEW = 4
};

bool IsCubic(te::Operation op, int substantial = SUBSTANTIAL);

bool IsAllred(te::Operation op, int substantial = SUBSTANTIAL);

bool IsShuffle(te::Operation op);

bool IsLocal(te::Operation op, int substantial = SUBSTANTIAL);

bool IsView(te::Operation op);

OpPattern GetOpPattern(te::Operation op);

} // namespace auto_tensorize

} // namespace ditto