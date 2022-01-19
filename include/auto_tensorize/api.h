#pragma once

#include <auto_compute/graph.h>
#include <auto_tensorize/state.h>
#include <hardware/hw_model.h>

namespace ditto {
using namespace auto_compute;
namespace auto_tensorize {

/*!
 * \brief A class for layer and schedule.
 */
class LayerAndScheduleNode : public Object {
public:
  /*! \brief The layer */
  Layer layer;
  /*! \brief The schedule */
  te::Schedule sch;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layer", &layer);
    v->Visit("sch", &sch);
  }

  static constexpr const char *_type_key =
      "ditto.auto_tensorize.LayerAndSchedule";
  TVM_DECLARE_BASE_OBJECT_INFO(LayerAndScheduleNode, Object);
};

class LayerAndSchedule : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param layer The operation
   * \param sch The schedule
   */
  TVM_DLL LayerAndSchedule(Layer layer, te::Schedule sch);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LayerAndSchedule, ObjectRef,
                                        LayerAndScheduleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerAndScheduleNode);
};

/*!
 * \brief The entry function of auto_schedule that uses auto_tensorize.
 * \param layer The layer to schedule
 * \param hw The hardware to schedule for
 */
TVM_DLL LayerAndSchedule auto_schedule(Layer layer, hardware::Hardware hw);

} // namespace auto_tensorize

} // namespace ditto