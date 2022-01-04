#include <hardware/visa/matrix_isa.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(MatrixISANode);

MatrixISA::MatrixISA(String name, double latency, te::Operation func) {
  auto node = make_object<MatrixISANode>();
  node->name = name;
  node->latency = latency;
  node->func = func;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.MatrixISA")
    .set_body_typed([](String name, double latency, te::Operation func) {
      return MatrixISA(name, latency, func);
    });

} // namespace hardware

} // namespace ditto