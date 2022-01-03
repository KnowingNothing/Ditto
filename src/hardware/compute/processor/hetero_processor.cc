#include <hardware/compute/processor/hetero_processor.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareProcessorNode);
TVM_REGISTER_NODE_TYPE(HeteroProcessorNode);

HeteroProcessor::HeteroProcessor(String name, Array<HardwareUnit> units,
                                 Array<LocalMemory> local_mems,
                                 Topology topology) {
  auto node = make_object<HeteroProcessorNode>();
  node->name = name;
  node->units = units;
  node->local_mems = local_mems;
  node->topology = topology;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HeteroProcessor")
    .set_body_typed([](String name, Array<HardwareUnit> units,
                       Array<LocalMemory> local_mems, Topology topology) {
      return HeteroProcessor(name, units, local_mems, topology);
    });

} // namespace hardware

} // namespace ditto