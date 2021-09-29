#pragma once

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>

using namespace tvm;
using namespace te;

namespace ditto {
namespace region {
// Node container for Region
class RegionNode;

class Region : public ObjectRef {
 public:
  Region() {}
  explicit Region(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a schedule for array of ops(and their dependencies).
   * \param ops The ops to be created as region.
   * \return sch The created Region.
   */
  TVM_DLL explicit Region(Operation op);

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const RegionNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline RegionNode* operator->();
  // declare container type
  using ContainerType = RegionNode;
};

/*! \brief node container for region */
class RegionNode : public Object {
 public:
  
  // contents here

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const char* _type_key = "Region";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegionNode, Object);
};

inline Region create_region(Operation op) { return Region(op); }

}  // namespace ditto
}  // namespace region
