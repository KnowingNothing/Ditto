#include <hardware/pattern/matrix_pattern.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(MatrixPatternNode);

MatrixPattern::MatrixPattern(String name, runtime::DataType dtype, int m, int n,
                             String qualifier) {
  auto node = make_object<MatrixPatternNode>();
  node->name = name;
  node->grain = te::placeholder({m, n}, dtype, "grain");
  node->qualifier = qualifier;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.MatrixPattern")
    .set_body_typed([](String name, runtime::DataType dtype, int m, int n,
                       String qualifier) {
      return MatrixPattern(name, dtype, m, n, qualifier);
    });

} // namespace hardware

} // namespace ditto