#include <hardware/pattern/matrix_pattern.h>

namespace ditto {

namespace hardware {

MatrixPattern::MatrixPattern(String name, runtime::DataType dtype, int m, int n,
                             String qualifier) {
  auto node = make_object<MatrixPatternNode>();
  node->name = name;
  node->grain = te::placeholder({m, n}, dtype, "grain");
  node->qualifier = qualifier;
  data_ = node;
}

} // namespace hardware

} // namespace ditto