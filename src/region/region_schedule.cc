#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <region/region_schedule.h>

#include <stack>
#include <unordered_set>

using namespace tvm;
using namespace te;

namespace ditto {
namespace region{

RegionSchedule::RegionSchedule(Array<Operation> ops) {
  auto n = make_object<RegionScheduleNode>();
  data_ = n;
  Schedule sch_(ops);
  n->sch = sch_;
}

TVM_REGISTER_NODE_TYPE(RegionScheduleNode);

TVM_REGISTER_GLOBAL("ditto.CreateRegionSchedule").set_body_typed(create_region_schedule);

TVM_REGISTER_GLOBAL("ditto.ScheduleNormalize").set_body_method(&Schedule::normalize);

TVM_REGISTER_GLOBAL("ditto.ScheduleCreateGroup").set_body_method(&Schedule::create_group);

TVM_REGISTER_GLOBAL("ditto.ScheduleCacheRead").set_body_method(&Schedule::cache_read);

TVM_REGISTER_GLOBAL("ditto.ScheduleCacheWrite").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[1].IsObjectRef<Tensor>()) {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Tensor(), args[2]);
  } else {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Array<Tensor>(), args[2]);
  }
});

TVM_REGISTER_GLOBAL("ditto.ScheduleRFactor").set_body_method(&Schedule::rfactor);

}  // namespace ditto
}  // namespace region
