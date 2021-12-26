#include <auto_tensorize/api.h>

namespace ditto {

namespace auto_tensorize {

LayerAndSchedule::LayerAndSchedule(Layer layer, te::Schedule sch) {
  auto node = make_object<LayerAndScheduleNode>();
  node->layer = layer;
  node->sch = sch;
  data_ = node;
}

std::pair<bool, std::string> validate(SerialFusionState state) {
  if (!state->IsLinearTopo()) {
    return std::make_pair(false, "the given layer is not in linear topology.");
  }
  int num_cubic_ops = state->CountOp(OpPattern::PATTERN_CUBIC);
  if (num_cubic_ops != 1 || num_cubic_ops != 2) {
    return std::make_pair(false,
                          "Ditto expect 1 or 2 cubic operators in the layer.");
  }
  int num_allred_ops = state->CountOp(OpPattern::PATTERN_ALLRED);
  if (num_allred_ops > 0) {
    return std::make_pair(
        false, ("auto_tensorize can't handle all reduce.\n The all reduce ops"
                "should be eliminated by the frontend.\n Did you forget to use"
                "Ditto's graph frontend?"));
  }
  return std::make_pair(true, "");
}

LayerAndSchedule auto_schedule(Layer layer, hardware::Hardware hw) {
  LayerAndSchedule ret;
  SerialFusionState state(layer);
  // first step, verify the layer pattern
  /*
   * Expected layer pattern:
   * 1. linear topology
   * 2. 1 or 2 cubic ops in the chain
   *    if 1 cubic op, only tensorize,
   *    if 2 cubic ops, fuse + tensorize
   * 3. no all_reduce op in the chain
   *    all reduce op should be handled
   *    by the front-end auto_compute.
   *    auto_compute will merge allred
   *    ops into cubic ops if possible.
   * 
   * We call this step 'validate'.
   */
  bool valid = false;
  std::string reason = "";
  std::tie(valid, reason) = validate(state);
  CHECK(valid) << "The given layer is not suitable for auto_tensorize because "
               << reason << "The layer is:\n" << layer << "\n.";
  // second step, match the cubic ops with the hardware units
  /*
   * The hardware provides heterogeneous compute units
   * including cube/matrix units and vector units,
   * so Ditto should check which compute units can be used.
   * For example, if the input is Conv+ReLU+Conv in 
   * mix precision and the hardware is V100 GPU with Tensor Core,
   * then Ditto will find that the first Conv can be mapped to
   * Tensor Core, the ReLU can be mapped to CUDA Core, and the
   * third Conv can be mapped to Tensor Core.
   * 
   * We call this step 'placement'.
   */
  return ret;
}

} // namespace auto_tensorize

} // namespace ditto