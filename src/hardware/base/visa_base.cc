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

} // namespace hardware

} // namespace ditto