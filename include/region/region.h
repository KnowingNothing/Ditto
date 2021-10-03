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
   * \brief Split the parent by factor
   * \param parent The parent iteration domain.
   * \param factor The split factor of the loop.
   * \param p_outer The result outer domain
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL Region& split_by_factor(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner);
  /*!
   * \brief Split the parent by nparts
   * \param parent The parent iteration domain.
   * \param nparts The number of parts in the outer domain.
   * \param p_outer The result outer domain.
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL Region& split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer, IterVar* p_inner);

  // Other schedule primitives here!

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
  /*! \brief Nodes in the iter var */
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

// implementations
inline const RegionNode* Region::operator->() const { return static_cast<const RegionNode*>(get()); }
inline RegionNode* Region::operator->() { return static_cast<RegionNode*>(get_mutable()); }

}  // namespace ditto
}  // namespace region
