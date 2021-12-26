#include <hardware/compute/unit/matrix_unit.h>

namespace ditto {

namespace hardware {

MatrixMultiplyAccumulate::MatrixMultiplyAccumulate(
    String name, double latency, Array<te::Operation> functionality) {
  auto node = make_object<MatrixMultiplyAccumulateNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

} // namespace hardware

} // namespace ditto