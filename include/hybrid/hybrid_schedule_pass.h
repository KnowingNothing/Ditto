#pragma once

#include <tvm/te/schedule.h>
#include <tvm/tir/function.h>
#include <hybrid/hybrid_schedule.h>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

/*!
 * \brief To automatically inline the element-wise operations.
 *
 * \param sch The hybrid_schedule to be inlined.
 */
void AutoInlineElemWise(HybridSchedule sch);

/*!
 * \brief To automatically inline the broadcast operations.
 *
 * \param sch The hybrid_schedule to be inlined.
 */
void AutoInlineBroarcast(HybridSchedule sch);

/*!
 * \brief To automatically inline operations with injective writes
 *   (i.e. writes without reduction or sequential loops). Note
 *   that in this case, guarantees about contiguity, transpose, stride,
 *   alignemnt and memory footprint in general do not hold.
 *
 * \param sch The hybrid_schedule to be inlined.
 */
TVM_DLL void AutoInlineInjective(HybridSchedule sch);

/*!
 * \brief Infer the bound of all iteration variables relates to the schedule.
 *
 * \param sch The root hybrid_schedule to infer all the bounds.
 * \return the result bound of the iteration Variable
 */
Map<IterVar, Range> InferBound(const HybridSchedule& sch);

/*!
 * \brief HybridSchedule s' dependent operations.
 *
 * \param s The hybrid_schedule to be realized
 * \param dom_map The domain of each iter vars.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1 during lowering.
 *                                This is a debug feature for dataflow/axis analysis.
 *                                Note: If this is true, The lowered IR may be incorrect,
 *                                because we will also delete the init part of reduction
 * \return the result Stmt
 */
Stmt ScheduleOps(HybridSchedule s, Map<IterVar, Range> dom_map, bool debug_keep_trivial_loop);

}  // namespace hybrid
}  // namespace ditto
