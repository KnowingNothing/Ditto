/* This file is for OpNode::BuildRealize and OpNode::BuildProvide*/
#pragma once

#include <tvm/te/operation.h>
#include <hybrid/hybrid_schedule.h>

using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{

Stmt BuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt BuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt BaseComputeOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt ComputeOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt ExternOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt ExternOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt HybridOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt HybridOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt PlaceholderOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt PlaceholderOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt ScanOpNodeBuildRealize(const HybridStage& stage, const std::unordered_map<IterVar, Range>& realize_map, const Stmt& body, String storage_scope = "");

Stmt ScanOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

Stmt TensorComputeOpNodeBuildProvide(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

} // namespace hybrid
} // namespace ditto
