#include <hardware/memory/local/register_array.h>

namespace ditto {

namespace hardware {

RegisterArray::RegisterArray(String name, double kb,
                             Array<te::Tensor> pattern) {
  auto node = make_object<RegisterArrayNode>();
  node->name = name;
  node->kb = kb;
  node->pattern = pattern;
  data_ = node;
}

} // namespace hardware

} // namespace ditto