#include <dmlc/thread_local.h>
#include <tvm/driver/driver_api.h>
#include <hybrid/driver_api.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <mutex>
#include <stack>

using namespace tvm;

namespace ditto{

tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape, DataType dtype, std::string name,
                                      int data_alignment, int offset_factor, bool compact) {
  DataType storage_dtype = (dtype == DataType::Bool() ? DataType::Int(8) : dtype);
  auto data = tir::Var(name, PointerType(PrimType(storage_dtype)));
  bool has_any = false;
  if (!compact) {
    for (const auto& it : shape) {
      if (it.as<tir::VarNode>()) {
        has_any = true;
        break;
      }
    }
  }
  tir::BufferType buffer_type = has_any ? tir::kAutoBroadcast : tir::kDefault;

  PrimExpr elem_offset;
  if (offset_factor != 0) {
    elem_offset = tir::Var(name + "_elem_offset", shape[0].dtype());
  } else {
    elem_offset = PrimExpr();
  }

  return tir::Buffer(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, data_alignment,
                     offset_factor, buffer_type);
}

void GetBinds(const Array<ObjectRef>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list) {
  *out_binds = binds;

  for (const ObjectRef& x : args) {
    if (const te::TensorNode* tensor_node = x.as<te::TensorNode>()) {
      te::Tensor x_ref = GetRef<te::Tensor>(tensor_node);
      if (out_binds->find(x_ref) == out_binds->end()) {
        tir::Buffer buf =
            BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype, x_ref->op->name, -1, 0, compact);
        out_binds->Set(x_ref, buf);
        out_arg_list->push_back(buf);
      } else {
        out_arg_list->push_back((*out_binds)[x_ref]);
      }
    } else if (x.as<te::BufferNode>() || x.as<tir::VarNode>()) {
      out_arg_list->push_back(x);
    } else {
      LOG(FATAL)
          << "Expected type of the elements of args to be te::Tensor, te::Buffer or tir::Var, "
          << "but got a " << x->GetTypeKey();
    }
  }
}

Array<tvm::transform::Pass> CreatePassList(bool disable_loop_partition) {
  tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();

  bool disable_vectorize = pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
  bool instrument_bound_checkers =
      pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

  // Get any user-added passes
  Array<Array<ObjectRef>> add_lower_pass =
      pass_ctx->GetConfig<Array<Array<ObjectRef>>>("tir.add_lower_pass", Array<Array<ObjectRef>>())
          .value();

  Array<tvm::transform::Pass> user_lower_phase0 = Array<tvm::transform::Pass>();
  Array<tvm::transform::Pass> user_lower_phase1 = Array<tvm::transform::Pass>();
  Array<tvm::transform::Pass> user_lower_phase2 = Array<tvm::transform::Pass>();
  Array<tvm::transform::Pass> user_lower_phase3 = Array<tvm::transform::Pass>();

  // phase pasees is of the form
  // [[phase_number, pass], [phase_number, pass]... ]
  for (Array<ObjectRef> phase_pass : add_lower_pass) {
    const IntImmNode* phase_num = phase_pass[0].as<IntImmNode>();
    ICHECK(phase_num)
        << "Expected the first entry in the inner Array of tir.add_lower_pass to be an integer";
    int phase_num_val = phase_num->value;

    CHECK_GE(phase_num_val, 0);

    const tvm::transform::PassNode* pass_node = phase_pass[1].as<tvm::transform::PassNode>();
    tvm::transform::Pass pass = GetRef<tvm::transform::Pass>(pass_node);
    // Copy the pass into the correct phase
    if (phase_num_val == 0) {
      user_lower_phase0.push_back(pass);
    } else if (phase_num_val == 1) {
      user_lower_phase1.push_back(pass);
    } else if (phase_num_val == 2) {
      user_lower_phase2.push_back(pass);
    } else if (phase_num_val >= 3) {
      user_lower_phase3.push_back(pass);
    }
  }

  // Construct the pass list, inserting the user provided passes at the end of the phase

  // PHASE 0
  Array<tvm::transform::Pass> pass_list = user_lower_phase0;

  // PHASE 1
  pass_list.push_back(tir::transform::InjectPrefetch());
  pass_list.push_back(tir::transform::TextureFlatten());
  pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  pass_list.push_back(tir::transform::LowerMatchBuffer());
  pass_list.push_back(tir::transform::FlattenBuffer());
  pass_list.push_back(tir::transform::UnifyThreadBinding());
  pass_list.push_back(tir::transform::BF16Legalize());
  pass_list.push_back(tir::transform::NarrowDataType(32));
  pass_list.push_back(tir::transform::Simplify());

  // Add user-defined phase-1 passes
  pass_list.insert(pass_list.end(), user_lower_phase1.begin(), user_lower_phase1.end());

  // PHASE 2
  if (!disable_loop_partition) {
    pass_list.push_back(tir::transform::LoopPartition());
  }

  pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
  pass_list.push_back(tir::transform::InjectVirtualThread());
  pass_list.push_back(tir::transform::InjectDoubleBuffer());
  pass_list.push_back(tir::transform::StorageRewrite());
  pass_list.push_back(tir::transform::UnrollLoop());

  // Add user-defined phase-2 passes
  pass_list.insert(pass_list.end(), user_lower_phase2.begin(), user_lower_phase2.end());

  // PHASE 3
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  pass_list.push_back(tir::transform::RewriteUnsafeSelect());
  pass_list.push_back(tir::transform::HoistIfThenElse());

  // Add user-defined phase-3 passes
  pass_list.insert(pass_list.end(), user_lower_phase3.begin(), user_lower_phase3.end());

  if (instrument_bound_checkers) {
    pass_list.push_back(tir::transform::InstrumentBoundCheckers());
  }
  return pass_list;
}

IRModule LowerWithPassList(IRModule mod, Array<tvm::transform::Pass> pass_list) {
  auto optimize = tvm::transform::Sequential(pass_list);
  mod = optimize(std::move(mod));
  return mod;
}
  
IRModule ScheduleToModule(hybrid::HybridSchedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds) {
  // Convert hybrid schedule to IRModule
  Array<ObjectRef> out_arg_list;
  tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();

  sch = sch.normalize();

  // Before TIR transformation.
  Map<tir::IterVar, Range> bounds = hybrid::InferBound(sch);
  tir::Stmt stmt = hybrid::ScheduleOps(sch, std::move(bounds), false);
  bool compact = te::VerifyCompactBuffer(stmt);

  Map<te::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, &out_arg_list);

  // Build the function
  // At this point binds is only te::Tensors
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  // Mark this hybrid schedule as being converted from an hybrid schedule. Makes sure that
  // the correct TE passes are run.
  f = WithAttr(std::move(f), "from_legacy_te_schedule", Bool(true));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();

  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  return IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
}

TVM_REGISTER_GLOBAL("ditto.hybrid_schedule_to_module")
    .set_body_typed([](hybrid::HybridSchedule sch, const Array<ObjectRef>& args, const String& name,
                       const Map<te::Tensor, tir::Buffer>& binds) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.get() != nullptr) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      IRModule mod = ScheduleToModule(std::move(sch), args, name, c_binds);
      return mod;
    });

IRModule LowerSchedule(hybrid::HybridSchedule sch, const Array<te::Tensor>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
  Array<ObjectRef> ref_args;
  for (ObjectRef x : args) {
    ref_args.push_back(x);
  }
  return LowerSchedule(std::move(sch), ref_args, name, binds);
}

IRModule LowerSchedule(hybrid::HybridSchedule sch, const Array<ObjectRef>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
  IRModule mod = ScheduleToModule(std::move(sch), args, name, binds);
  // Get the legacy TE pass list
  Array<tvm::transform::Pass> pass_list = CreatePassList(simple_mode);
  return LowerWithPassList(mod, pass_list);
}

TVM_REGISTER_GLOBAL("ditto.lower_hybrid_schedule")
    .set_body_typed([](hybrid::HybridSchedule sch, const Array<ObjectRef>& args, const String& name,
                       const Map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.get() != nullptr) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      return LowerSchedule(std::move(sch), args, name, c_binds, simple_mode);
    });

} // namespace ditto
