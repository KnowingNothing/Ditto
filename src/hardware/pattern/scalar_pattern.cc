#include <hardware/pattern/scalar_pattern.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(ScalarPatternNode);

ScalarPattern::ScalarPattern(String name, runtime::DataType dtype,
                             String qualifier) {
  auto node = make_object<ScalarPatternNode>();
  node->name = name;
  node->grain = te::placeholder({1}, dtype, "grain");
  node->qualifier = qualifier;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.ScalarPattern")
    .set_body_typed([](String name, runtime::DataType dtype, String qualifier) {
      return ScalarPattern(name, dtype, qualifier);
    });

} // namespace hardware

} // namespace ditto