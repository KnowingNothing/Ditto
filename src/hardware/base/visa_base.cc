#include <hardware/base/visa_base.h>

namespace ditto {

namespace hardware {

ISA::ISA(String name, double latency, te::Operation func) {
  auto node = make_object<ISANode>();
  node->name = name;
  node->latency = latency;
  node->func = func;
  data_ = node;
}

tir::IterVar SpatialAxis(int extent, std::string name) {
  Range dom(0, extent);
  return tir::IterVar(dom, tir::Var(name, runtime::DataType::Int(32)),
                      tir::IterVarType::kDataPar);
}

tir::IterVar ReduceAxis(int extent, std::string name) {
  Range dom(0, extent);
  return tir::IterVar(dom, tir::Var(name, runtime::DataType::Int(32)),
                      tir::IterVarType::kCommReduce);
}

ISA Direct() { return ISA("special.direct", 0.0, te::Operation()); }

ISA None() { return ISA("special.none", 0.0, te::Operation()); }

} // namespace hardware

} // namespace ditto