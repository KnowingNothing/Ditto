#pragma once

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>

#include <string>
#include <unordered_map>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

// Node container for Schedule
class HybridScheduleNode;

/*!
 * \brief Global hybrid_schedule container
 */
class HybridSchedule : public ObjectRef {
 public:
  HybridSchedule() {}
  explicit HybridSchedule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a hybrid_schedule for array of ops(and their dependencies).
   * \param ops The ops to be scheduled.
   * \return sch The created HybridSchedule.
   */
  TVM_DLL explicit HybridSchedule(Array<Operation> ops);

  // new schedule primitives here!

  TVM_DLL HybridSchedule& slice(Tensor tensor, IterVar axis, PrimExpr slice_point, Tensor* tensor_l, Tensor* tensor_r);


  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const HybridScheduleNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline HybridScheduleNode* operator->();
  // declare container type
  using ContainerType = HybridScheduleNode;
};

/*! \brief node container for hybrid_schedule */
class HybridScheduleNode : public Object {
 public:
  Schedule sch;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("sch", &sch);
  }

  static constexpr const char* _type_key = "HybridSchedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(HybridScheduleNode, Object);
};

/*!
 * \brief Create a hybrid_schedule for array of ops(and their dependencies).
 * \param ops The ops to be scheduled.
 * \return sch The created HybridSchedule.
 */
inline HybridSchedule create_hybrid_schedule(Array<Operation> ops) { return HybridSchedule(ops); }

// implementations
inline const HybridScheduleNode* HybridSchedule::operator->() const {
  return static_cast<const HybridScheduleNode*>(get());
}
inline HybridScheduleNode* HybridSchedule::operator->() { return static_cast<HybridScheduleNode*>(get_mutable()); }

}  // namespace ditto
}  // namespace hybrid
