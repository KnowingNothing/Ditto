#include <hardware/compute/processor/hetero_processor.h>

namespace ditto {

namespace hardware {

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

} // namespace hardware

} // namespace ditto