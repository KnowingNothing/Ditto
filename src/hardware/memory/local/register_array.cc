#include <hardware/memory/local/register_array.h>

namespace ditto {

namespace hardware {

RegisterArray::RegisterArray(String name, double kb,
                             Map<String, Pattern> pattern_list) {
  auto node = make_object<RegisterArrayNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto