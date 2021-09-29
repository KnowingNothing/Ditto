#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <region/region.h>

#include <stack>
#include <unordered_set>

using namespace tvm;
using namespace te;

namespace ditto {
namespace region {

Region::Region(Operation op) {
  auto n = make_object<RegionNode>();
  n->op = op;
  n->iter_vars = op->root_iter_vars();
  // remove opaque var from leaf.
  Array<IterVar> clean;
  for (IterVar iv : n->iter_vars) {
    if (iv->iter_type != kOpaque) clean.push_back(iv);
  }
  n->iter_vars = clean;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(RegionNode);

TVM_REGISTER_GLOBAL("ditto.CreateRegion").set_body_typed(create_region);

}  // namespace ditto
}  // namespace region
