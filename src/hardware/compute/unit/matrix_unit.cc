#include <hardware/compute/unit/matrix_unit.h>

namespace ditto {

namespace hardware {

MatrixMultiplyAccumulate::MatrixMultiplyAccumulate(String name,
                                                   Map<String, ISA> isa_list) {
  auto node = make_object<MatrixMultiplyAccumulateNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto