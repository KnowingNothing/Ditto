#pragma once

#include <tvm/te/schedule.h>
#include <hybrid/hybrid_schedule.h>
#include <tvm/tir/expr.h>
#include "../build_for_ops.h"

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

/*!
 * \brief Apply the schedule manipulation on the function body.
 * \param stmt The statement to be processed.
 * \param dom_map The extents of the iterative variables may be used.
 * \param stage The schedule information to be applied.
 */
Stmt ApplySchedule(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                   Stmt stmt);

/*!
 * \brief Apply loop splits and fuses in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param dom_map The extents of the iterative variables may be used.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopShapes(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     Stmt stmt);

/*!
 * \brief Apply loop annotation in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param rebased The map specifies the rebase, a.k.a rename, relationship of these variables.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopAnnotations(const HybridStage& stage, const std::unordered_map<IterVar, IterVar>& rebased,
                          Stmt stmt);

/*!
 * \brief Apply loop order in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param dom_map The extents of the iterative variables may be used.
 * \param rebased The map specifies the rebase, a.k.a rename, relationship of these variables.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopOrder(const HybridStage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<IterVar, IterVar>& rebased, Stmt stmt);

}  // namespace hybrid
}  // namespace ditto
