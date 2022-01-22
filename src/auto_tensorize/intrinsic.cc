#include <auto_tensorize/intrinsic.h>

namespace ditto {

namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(PackedIntrinsicNode);

PackedIntrinsic::PackedIntrinsic(Array<Intrinsic> load_intrinsics,
                                 Intrinsic compute_intrinsic,
                                 Intrinsic store_intrinsic) {
  auto node = make_object<PackedIntrinsicNode>();
  node->load_intrinsics = load_intrinsics;
  node->compute_intrinsic = compute_intrinsic;
  node->store_intrinsic = store_intrinsic;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.PackedIntrinsic")
    .set_body_typed([](Array<Intrinsic> load_intrinsics,
                       Intrinsic compute_intrinsic, Intrinsic store_intrinsic) {
      return PackedIntrinsic(load_intrinsics, compute_intrinsic,
                             store_intrinsic);
    });

} // namespace auto_tensorize

} // namespace ditto