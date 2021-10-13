#pragma once

#include "../build_for_ops.h"
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_map>
#include <vector>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {
// loop nest structure for general compute
// This the loop nest structured used in compute.
// Does not include the loop body.
struct ComputeLoopNest {
  // The common number of loops between init and main
  size_t num_common_loop;
  // predicates for the initialize loop
  std::vector<PrimExpr> init_predicates;
  // Initialization nest involved.
  std::vector<std::vector<Stmt> > init_nest;
  // Value map for the init code
  std::unordered_map<IterVar, PrimExpr> init_vmap;
  // Predicates for the main update loop
  std::vector<PrimExpr> main_predicates;
  // The general loop nest
  std::vector<std::vector<Stmt> > main_nest;
  // Value map for the IterVar.
  std::unordered_map<IterVar, PrimExpr> main_vmap;

  /*!
   * \brief constructor to build ComputeOpNest
   * \param self The pointer to compute op.
   * \param stage The scxhedule stage.
   * \param dom_map The domain map.
   * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
   * \return The constructed loop nest
   */
  static ComputeLoopNest Create(const BaseComputeOpNode* self, const HybridStage& stage,
                                const std::unordered_map<IterVar, Range>& dom_map,
                                bool debug_keep_trivial_loop);
};

/*!
 * \brief Build body of compute for cross thread reduction pattern.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeCrossThreadReduction(const ComputeOpNode* self, const HybridStage& stage,
                              const std::unordered_map<IterVar, Range>& dom_map,
                              bool debug_keep_trivial_loop);

/*!
 * \brief Build body of compute for tensorization.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeTensorize(const ComputeOpNode* self, const HybridStage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

/*!
 * \brief Transform the update part when there is no init func in tensorizing
 * \param stage The stage for tensorizing.
 * \param dom_map The range of each iter var.
 * \param n The loop nest structured used in compute.
 * \param body The body func in tensorize intrin
 * \param update The update func in tensorize intrin
 * \return Transformed result.
 */
Stmt TransformUpdate(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n, Stmt body, Stmt update);
}  // namespace hybrid
}  // namespace ditto
