#include <auto_tensorize/intrinsic.h>

namespace ditto {

namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(PackedIntrinsicNode);

PackedIntrinsic::PackedIntrinsic(Array<Intrinsic> load_intrinsics,
                                 Intrinsic compute_intrinsic,
                                 Intrinsic store_intrinsic,
                                 Array<String> load_scopes,
                                 String compute_scope, String store_scope) {
  auto node = make_object<PackedIntrinsicNode>();
  node->load_intrinsics = load_intrinsics;
  node->compute_intrinsic = compute_intrinsic;
  node->store_intrinsic = store_intrinsic;
  node->load_scopes = load_scopes;
  node->compute_scope = compute_scope;
  node->store_scope = store_scope;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.PackedIntrinsic")
    .set_body_typed([](Array<Intrinsic> load_intrinsics,
                       Intrinsic compute_intrinsic, Intrinsic store_intrinsic,
                       Array<String> load_scopes, String compute_scope,
                       String store_scope) {
      return PackedIntrinsic(load_intrinsics, compute_intrinsic,
                             store_intrinsic, load_scopes, compute_scope,
                             store_scope);
    });

} // namespace auto_tensorize

} // namespace ditto