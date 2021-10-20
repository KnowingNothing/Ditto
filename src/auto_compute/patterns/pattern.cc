#include <auto_compute/patterns/pattern.h>

namespace ditto {

namespace auto_compute {

TVM_REGISTER_NODE_TYPE(PatternNode);

TVM_DLL Pattern::Pattern(Array<IntImm> tensor_ids,
                         Array<Array<IntImm>> iter_ids_array) {
  auto node = make_object<PatternNode>();
  node->tensor_ids = tensor_ids;
  node->iter_ids_array = iter_ids_array;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_compute.MakePattern")
    .set_body_typed([](Array<IntImm> tensor_ids,
                       Array<Array<IntImm>> iter_ids_array) {
      return Pattern(tensor_ids, iter_ids_array);
    });

} // namespace auto_compute

} // namespace ditto