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

Region& Region::split_by_factor(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner) {
  // todo: implement this schedule primitive
  std::cout<<"in split_by_factor"<<std::endl;
  return *this;
}

Region& Region::split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer, IterVar* p_inner) {
  // todo: implement this schedule primitive
  std::cout<<"in split_by_nparts"<<std::endl;
  return *this;
}

TVM_REGISTER_NODE_TYPE(RegionNode);

TVM_REGISTER_GLOBAL("ditto.CreateRegion").set_body_typed(create_region);
TVM_REGISTER_GLOBAL("ditto.RegionSplitByFactor")
    .set_body_typed([](Region region, IterVar parent, PrimExpr factor) {
      IterVar outer, inner;
      region.split_by_factor(parent, factor, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("ditto.RegionSplitByNParts")
    .set_body_typed([](Region region, IterVar parent, PrimExpr nparts) {
      IterVar outer, inner;
      region.split_by_nparts(parent, nparts, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

}  // namespace ditto
}  // namespace region
