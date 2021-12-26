#include <hardware/memory/local/fragment.h>

namespace ditto {

namespace hardware {

Fragment::Fragment(String name, double kb, Array<te::Tensor> pattern) {
  auto node = make_object<FragmentNode>();
  node->name = name;
  node->kb = kb;
  node->pattern = pattern;
  data_ = node;
}

} // namespace hardware

} // namespace ditto