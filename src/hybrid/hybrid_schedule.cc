#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <hybrid/hybrid_schedule.h>

#include <stack>
#include <unordered_set>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid{

HybridSchedule::HybridSchedule(Array<Operation> ops) {
  auto n = make_object<HybridScheduleNode>();
  data_ = n;
  Schedule sch_(ops);
  n->sch = sch_;
}

HybridSchedule& HybridSchedule::slice(Tensor tensor, IterVar axis, PrimExpr slice_point, Tensor* tensor_l, Tensor* tensor_r){
  // search Schedule::cache_read to see how to create a new Tensor
  std::cout<<"in slice"<<std::endl;
  return *this;
}

TVM_REGISTER_NODE_TYPE(HybridScheduleNode);

TVM_REGISTER_GLOBAL("ditto.CreateHybridSchedule").set_body_typed(create_hybrid_schedule);

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

TVM_REGISTER_GLOBAL("ditto.HybridScheduleSlice")
    .set_body_typed([](HybridSchedule rsch, Tensor tensor, IterVar axis, PrimExpr slice_point) {
      Tensor tensor_l, tensor_r;
      rsch.slice(tensor, axis, slice_point, &tensor_l, &tensor_r);
      return Array<Tensor>({tensor_l, tensor_r});
    });

}  // namespace ditto
}  // namespace hybrid
