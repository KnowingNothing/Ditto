#include <hardware/pattern/scalar_pattern.h>

namespace ditto {

namespace hardware {

ScalarPattern::ScalarPattern(String name, runtime::DataType dtype,
                             String qualifier) {
  auto node = make_object<ScalarPatternNode>();
  node->name = name;
  node->grain = te::placeholder({1}, dtype, "grain");
  node->qualifier = qualifier;
  data_ = node;
}

} // namespace hardware

} // namespace ditto