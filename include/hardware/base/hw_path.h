#pragma once

#include <hardware/base/hw_base.h>
#include <hardware/base/pattern_base.h>
#include <hardware/base/visa_base.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for path.
 */
class HardwarePathNode : public Object {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {}
  static constexpr const char *_type_key = "ditto.hardware.HardwarePath";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwarePathNode, Object);
};

class HardwarePath : public ObjectRef {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwarePath, ObjectRef,
                                        HardwarePathNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwarePathNode);
};

/*!
 * \brief class for compute path.
 */
class ComputePathNode : public HardwarePathNode {
public:
  /*! \brief The virtual isa */
  ISA isa;
  /*! \brief The pattern */
  Pattern pattern;
  /*! \brief The load isa */
  ISA load;
  /*! \brief The store isa */
  ISA store;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("isa", &isa);
    v->Visit("pattern", &pattern);
    v->Visit("load", &load);
    v->Visit("store", &store);
  }

  static constexpr const char *_type_key = "ditto.hardware.ComputePath";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputePathNode, Object);
};

class ComputePath : public HardwarePath {
public:
  /*!
   * \brief The constructor.
   * \param isa The compute isa
   * \param pattern The memory pattern
   * \param load The load isa
   * \param store The store isa
   */
  TVM_DLL ComputePath(ISA isa, Pattern pattern, ISA load, ISA store);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ComputePath, HardwarePath,
                                        ComputePathNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputePathNode);
};

/*!
 * \brief class for data path.
 */
class DataPathNode : public HardwarePathNode {
public:
  /*! \brief The virtual isa */
  ISA isa;
  /*! \brief The pattern of the source*/
  Pattern src_pattern;
  /*! \brief The pattern of the dst*/
  Pattern dst_pattern;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("isa", &isa);
    v->Visit("src_pattern", &src_pattern);
    v->Visit("dst_pattern", &dst_pattern);
  }

  static constexpr const char *_type_key = "ditto.hardware.DataPath";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataPathNode, Object);
};

class DataPath : public HardwarePath {
public:
  /*!
   * \brief The constructor.
   * \param isa The compute isa
   * \param src_pattern The memory pattern of the source
   * \param dst_pattern The memory pattern of the dst
   */
  TVM_DLL DataPath(ISA isa, Pattern src_pattern, Pattern dst_pattern);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DataPath, HardwarePath, DataPathNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataPathNode);
};

} // namespace hardware

} // namespace ditto