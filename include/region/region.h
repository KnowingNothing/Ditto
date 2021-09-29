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
  /*!
   * \brief The operation of region.
   */
  Operation op;
  /*! \brief All the nodes in the iter var */
  Array<IterVar> iter_vars;
  /*!
   * \brief Subregions of this region.
   * Used to store regions generated after slicing.
   */
  Array<Region> subregions;
  /*! \brief inidicate whether this region is leaf */
  bool is_leaf{true};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("iter_vars", &iter_vars);
    v->Visit("subregions", &subregions);
    v->Visit("is_leaf", &is_leaf);
  }

  static constexpr const char* _type_key = "Region";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegionNode, Object);
};

inline Region create_region(Operation op) { return Region(op); }

}  // namespace ditto
}  // namespace region
