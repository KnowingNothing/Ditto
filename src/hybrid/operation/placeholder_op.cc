#include "../build_for_ops.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

Stmt PlaceholderOpNodeBuildRealize(const HybridStage& stage,
                                     const std::unordered_map<IterVar, Range>& realize_map,
                                     const Stmt& body, String storage_scope){
  return body;
}

Stmt PlaceholderOpNodeBuildProvide(const HybridStage& stage,
                                     const std::unordered_map<IterVar, Range>& dom_map,
                                     bool debug_keep_trivial_loop){
  return Stmt();
}
}  // namespace hybrid
}  // namespace ditto
