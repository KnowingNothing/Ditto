#pragma once

#include <hardware/base/pattern_base.h>

using namespace tvm;
namespace ditto {

namespace hardware {

enum class MatrixLayout : int { RowMajor = 0, ColMajor = 1 };

/*!
 * \brief A base class for matrix access pattern.
 */
class MatrixPatternNode : public PatternNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("grain", &grain);
    v->Visit("qualifier", &qualifier);
  }

  static constexpr const char *_type_key = "ditto.hardware.MatrixPattern";
  TVM_DECLARE_BASE_OBJECT_INFO(MatrixPatternNode, Object);
};

class MatrixPattern : public Pattern {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the pattern
   * \param dtype The data type of the pattern
   * \param m The matrix heigh
   * \param n The matrix width
   * \param qualifier The qualifier of the pattern
   */
  TVM_DLL MatrixPattern(String name, runtime::DataType dtype, int m, int n,
                        String qualifier);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MatrixPattern, Pattern,
                                        MatrixPatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatrixPatternNode);
};

} // namespace hardware

} // namespace ditto