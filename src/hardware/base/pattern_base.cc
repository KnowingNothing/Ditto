#include <hardware/base/pattern_base.h>

namespace ditto {

namespace hardware {

Pattern::Pattern(String name, te::Tensor grain, String qualifier) {
  auto node = make_object<PatternNode>();
  node->name = name;
  node->grain = grain;
  node->qualifier = qualifier;
  data_ = node;
}

} // namespace hardware

} // namespace ditto