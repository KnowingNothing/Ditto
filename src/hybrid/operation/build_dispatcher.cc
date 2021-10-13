/* This file is for OpNode::BuildRealize and OpNode::BuildProvide*/
#include "../build_for_ops.h"

using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{

Stmt BuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope){
  if(stage->op->IsInstance<ExternOpNode>())
    return ExternOpNodeBuildRealize(stage, realize_map, body, storage_scope);
  else if(stage->op->IsInstance<HybridOpNode>()) 
    return HybridOpNodeBuildRealize(stage, realize_map, body, storage_scope);
  else if(stage->op->IsInstance<PlaceholderOpNode>()) 
    return PlaceholderOpNodeBuildRealize(stage, realize_map, body, storage_scope);
  else if(stage->op->IsInstance<ScanOpNode>()) 
    return ScanOpNodeBuildRealize(stage, realize_map, body, storage_scope);
  else // if(stage->op->IsInstance<ComputeOpNode>() or stage->op->IsInstance<TensorComputeOpNode>())
    return BaseComputeOpNodeBuildRealize(stage, realize_map, body, storage_scope);
}

Stmt BuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop){
  if(stage->op->IsInstance<ExternOpNode>())
    return ExternOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
  else if(stage->op->IsInstance<HybridOpNode>()) 
    return HybridOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
  else if(stage->op->IsInstance<PlaceholderOpNode>()) 
    return PlaceholderOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
  else if(stage->op->IsInstance<ScanOpNode>()) 
    return ScanOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
  else if(stage->op->IsInstance<TensorComputeOpNode>()) 
    return TensorComputeOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
  else // if(stage->op->IsInstance<ComputeOpNode>())
    return ComputeOpNodeBuildProvide(stage, dom_map, debug_keep_trivial_loop);
}

} // namespace hybrid
} // namespace ditto
