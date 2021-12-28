#pragma once

#include <hardware/base/hw_path.h>
#include <hardware/compute/processor/hw_processor.h>
#include <hardware/compute/unit/hw_unit.h>
#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

using Topology = Map<Hardware, Map<LocalMemory, HardwarePath>>;

/*!
 * \brief A class for heterogeneous processor.
 */
class HeteroProcessorNode : public HardwareProcessorNode {
public:
  /*! \brief The hardware units */
  Array<HardwareUnit> units;
  /*! \brief The local memory list */
  Array<LocalMemory> local_mems;
  /*! \brief The topology */
  Topology topology;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("units", &units);
    v->Visit("local_mems", &local_mems);
    v->Visit("topology", &topology);
  }

  static constexpr const char *_type_key = "ditto.hardware.HeteroProcessor";
  TVM_DECLARE_FINAL_OBJECT_INFO(HeteroProcessorNode, HardwareProcessorNode);
};

class HeteroProcessor : public HardwareProcessor {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param units The units in this processor
   * \param local_mems The local memory
   * \param topology The topology of units and memory
   */
  TVM_DLL HeteroProcessor(String name, Array<HardwareUnit> units,
                          Array<LocalMemory> local_mems, Topology topology);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HeteroProcessor, HardwareProcessor,
                                        HeteroProcessorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HeteroProcessorNode);
};

} // namespace hardware

} // namespace ditto