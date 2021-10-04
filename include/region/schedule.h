#pragma once

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>

namespace ditto {
namespace region {

class Schedule : public tvm::te::Schedule{
};

class ScheduleNode : public tvm::te::ScheduleNode {
};

inline Schedule create_schedule(Array<Operation> ops) { return Schedule(ops); }

}  // namespace ditto
}  // namespace region