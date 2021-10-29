#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <hybrid/hybrid_schedule.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ditto {
namespace hybrid {
/*!
 * \brief Downward inference of domain of each IterVar.
 *  Caller set the range of the root, then the function
 *  propagates it towards the leaves.
 *
 * \param stage The hybrid_stage to operate on.
 * \param p_state The state of the message passing.
 * \param analyzer Analyzer context, storing information about bounds in p_state.
 * \param allow_missing Whether allow missing value.
 */
void PassDownDomain(const HybridStage& stage, std::unordered_map<IterVar, Range>* p_state,
                    arith::Analyzer* analyzer, bool allow_missing = false);

/*!
 * \param Upward inference of index of each IterVar.
 *  given index assignement of the leaves,
 *
 * \param stage The hybrid_stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpIndex(const HybridStage& stage, const Map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing = false);

/*!
 * \param Downward inference of index of each IterVar.
 *  given index assignement of roots.
 *
 * \param stage The hybrid_stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownIndex(const HybridStage& stage, const Map<IterVar, Range>& dom_map,
                   std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing = false);

/*!
 * \param Upward inference of domain set of each IterVar.
 *  given domain assignment of the leaves,
 *
 * \param stage The hybrid_stage to operate on.
 * \param dom_map The domain map of each iteration variable's maximum domain.
 * \param p_state The index state of each IterVar.
 */
void PassUpDomain(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, IntSet>* p_state);

/*!
 * \brief Upward message passing of bitmask with or relation.
 * \param stage The hybrid_stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpBitMaskOr(const HybridStage& stage, std::unordered_map<IterVar, int>* p_state,
                     bool allow_missing = false);

/*!
 * \brief Downward message passing of bitmask with or relation.
 * \param stage The hybrid_stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownBitMaskOr(const HybridStage& stage, std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing = false);

void PassDownBitMaskOr_WithoutSlice(const HybridStage& stage, std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing = false);

/*!
 * \brief Create boundary check predicates given remapped value of root
 * \param stage The hybrid_stage we operate on
 * \param dom_map The domain map of each value.
 * \param value_map The value map of the root iter var.
 * \param skip_ivar_domain Whether we skip check for IterVar's original domain.
 * \param skip_iter The set of variables to skip bound condition.
 * \return List of predicates that we need to check.
 */
std::vector<PrimExpr> MakeBoundCheck(const HybridStage& stage, const Map<IterVar, Range>& dom_map,
                                     const std::unordered_map<IterVar, PrimExpr>& value_map,
                                     bool skip_ivar_domain,
                                     const std::unordered_set<IterVar>& skip_iter);

std::vector<PrimExpr> MakeBoundCheck_WithIter(const HybridStage& stage, const Map<IterVar, Range>& dom_map,
                                     const std::unordered_map<IterVar, PrimExpr>& value_map,
                                     bool skip_ivar_domain,
                                     const std::unordered_set<IterVar>& skip_iter,
                                     std::vector<IterVar>& iters);

}  // namespace hybrid
}  // namespace ditto
