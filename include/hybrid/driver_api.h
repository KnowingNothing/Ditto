#pragma once

#include <tvm/ir/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/function.h>
#include <hybrid/hybrid_schedule_pass.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace tvm;

namespace ditto{

/*!
 * \brief Build an IRModule given a hybrid schedule, args and binds. This function also applies
 * the lowering passes defined in CreatePassList.
 * \param sch The hybrid schedule to lower.
 * \param args The arguments to the function.
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */

TVM_DLL IRModule LowerSchedule(hybrid::HybridSchedule sch, 
                               const Array<te::Tensor>& args,
                               const std::string& name,
                               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                               bool simple_mode = false);

/*!
 * \brief Build an IRModule given a hybrid schedule, args and binds. This function also applies
 * the lowering passes defined in CreatePassList.
 * \param sch The hybrid schedule to lower.
 * \param args The arguments to the function (Array of Tensor, Buffer and Vars)
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */
TVM_DLL IRModule LowerSchedule(hybrid::HybridSchedule sch, const Array<ObjectRef>& args,
                               const std::string& name,
                               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                               bool simple_mode = false);

/*!
 * \brief Create an IRModule out of a hybrid Schedule. It does not apply lowering passes. If you want
 * to apply lowering passes as well, use LowerSchedule.
 * \param sch The hybrid schedule
 * \param args The arguments to the function.
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \return The result module.
 */
IRModule ScheduleToModule(hybrid::HybridSchedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds);
                          
} // namespace ditto
