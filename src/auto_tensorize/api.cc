#include <auto_tensorize/api.h>

namespace ditto {

namespace auto_tensorize {

LayerAndSchedule::LayerAndSchedule(Layer layer, te::Schedule sch) {
  auto node = make_object<LayerAndScheduleNode>();
  node->layer = layer;
  node->sch = sch;
  data_ = node;
}

} // namespace auto_tensorize

} // namespace ditto