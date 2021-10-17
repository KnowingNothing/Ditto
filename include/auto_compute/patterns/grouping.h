#pragma once

#include <auto_compute/patterns/pattern.h>
#include <auto_compute/patterns/utils.h>
#include <tvm/tir/expr.h>

namespace ditto {
using namespace tvm;
namespace auto_compute {

Array<Pattern> FindGroupingPattern(const te::Operation &op);

} // namespace auto_compute

} // namespace ditto