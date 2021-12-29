#pragma once

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for virtual isa.
 */
class ISANode : public Object {
public:
  /*! \brief The name of this isa */
  String name;
  /*! \brief The latency of this isa */
  double latency;
  /*! \brief The functionality of the isa */
  te::Operation func;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("func", &func);
  }

  static constexpr const char *_type_key = "ditto.hardware.ISA";
  TVM_DECLARE_BASE_OBJECT_INFO(ISANode, Object);
};

class ISA : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the isa
   * \param latency The latency of the isa
   * \param func The functionality of the isa
   */
  TVM_DLL ISA(String name, double latency, te::Operation func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ISA, ObjectRef, ISANode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ISANode);
};

tir::IterVar SpatialAxis(int extent, std::string name = "Siv");

tir::IterVar ReduceAxis(int extent, std::string name = "Riv");

ISA Direct();

ISA None();

} // namespace hardware

} // namespace ditto