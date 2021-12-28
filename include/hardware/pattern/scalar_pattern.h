#pragma once

#include <hardware/base/pattern_base.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for scalar access pattern.
 */
class ScalarPatternNode : public PatternNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("grain", &grain);
    v->Visit("qualifier", &qualifier);
  }

  static constexpr const char *_type_key = "ditto.hardware.ScalarPattern";
  TVM_DECLARE_BASE_OBJECT_INFO(ScalarPatternNode, Object);
};

class ScalarPattern : public Pattern {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the pattern
   * \param dtype The data type of the pattern
   * \param qualifier The qualifier of the pattern
   */
  TVM_DLL ScalarPattern(String name, runtime::DataType dtype, String qualifier);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScalarPattern, Pattern,
                                        ScalarPatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScalarPatternNode);
};

} // namespace hardware

} // namespace ditto