#pragma once

#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <hybrid/hybrid_schedule.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../../3rdparty/tvm/src/tir/transforms/arg_binder.h"
#include "../../../3rdparty/tvm/src/tir/transforms/ir_utils.h"
#include "../message_passing.h"

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

using tir::MergeNest;

/*!
 * \brief Build loop nest for stage.
 *
 * \param stage The stage to create a loop nest.
 * \param dom_map The range of each iter var.
 * \param begin_iter_pos The beginning position of leaf_iter_vars to generate loop.
 * \param new_loop_var Whether create new loop variable.
 * \param skip_iter Whether skip certain iteration.
 * \param p_value_map The result value of each IterVar.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 */
std::vector<std::vector<Stmt> > MakeLoopNest(const HybridStage& stage,
                                             const std::unordered_map<IterVar, Range>& dom_map,
                                             size_t begin_iter_pos, bool new_loop_var,
                                             const std::unordered_set<IterVar>& skip_iter,
                                             std::unordered_map<IterVar, PrimExpr>* p_value_map,
                                             bool debug_keep_trivial_loop);

}  // namespace hybrid
}  // namespace ditto
