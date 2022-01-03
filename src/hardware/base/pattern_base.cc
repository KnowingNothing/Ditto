#include <hardware/base/pattern_base.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(PatternNode);

Pattern::Pattern(String name, te::Tensor grain, String qualifier) {
  auto node = make_object<PatternNode>();
  node->name = name;
  node->grain = grain;
  node->qualifier = qualifier;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.Pattern")
    .set_body_typed([](String name, te::Tensor grain, String qualifier) {
      return Pattern(name, grain, qualifier);
    });

} // namespace hardware

} // namespace ditto