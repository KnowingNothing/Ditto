#include <hardware/compute/unit/scalar_unit.h>

namespace ditto {

namespace hardware {

Adder::Adder(String name, double latency, Array<te::Operation> functionality) {
  auto node = make_object<AdderNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

Multiplier::Multiplier(String name, double latency,
                       Array<te::Operation> functionality) {
  auto node = make_object<MultiplierNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

MultiplyAdder::MultiplyAdder(String name, double latency,
                             Array<te::Operation> functionality) {
  auto node = make_object<MultiplyAdderNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

} // namespace hardware

} // namespace ditto