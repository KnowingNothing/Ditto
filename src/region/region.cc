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

Region::Region(Array<Operation> ops) {
    std::cout<<"hello"<<std::endl;
}

TVM_REGISTER_NODE_TYPE(RegionNode);

TVM_REGISTER_GLOBAL("ditto.CreateRegion").set_body_typed(create_region);

}  // namespace ditto
}  // namespace region
