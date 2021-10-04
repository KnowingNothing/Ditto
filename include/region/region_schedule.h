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
namespace region {

// Node container for Schedule
class RegionScheduleNode;

/*!
 * \brief Global region_schedule container
 */
class RegionSchedule : public ObjectRef {
 public:
  RegionSchedule() {}
  explicit RegionSchedule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a region_schedule for array of ops(and their dependencies).
   * \param ops The ops to be scheduled.
   * \return sch The created RegionSchedule.
   */
  TVM_DLL explicit RegionSchedule(Array<Operation> ops);

  // new schedule primitives here!

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const RegionScheduleNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline RegionScheduleNode* operator->();
  // declare container type
  using ContainerType = RegionScheduleNode;
};

/*! \brief node container for region_schedule */
class RegionScheduleNode : public Object {
 public:
  Schedule sch;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("sch", &sch);
  }

  static constexpr const char* _type_key = "RegionSchedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegionScheduleNode, Object);
};

/*!
 * \brief Create a region_schedule for array of ops(and their dependencies).
 * \param ops The ops to be scheduled.
 * \return sch The created RegionSchedule.
 */
inline RegionSchedule create_region_schedule(Array<Operation> ops) { return RegionSchedule(ops); }

// implementations
inline const RegionScheduleNode* RegionSchedule::operator->() const {
  return static_cast<const RegionScheduleNode*>(get());
}
inline RegionScheduleNode* RegionSchedule::operator->() { return static_cast<RegionScheduleNode*>(get_mutable()); }

}  // namespace ditto
}  // namespace region
