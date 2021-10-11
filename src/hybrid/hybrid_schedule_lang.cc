#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <hybrid/hybrid_schedule.h>

#include "graph.h"

#include <stack>
#include <unordered_set>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid{

Stage CopyStage(const Stage& s) {
  ObjectPtr<StageNode> n = make_object<StageNode>(*s.operator->());
  return Stage(n);
}

HybridSchedule HybridSchedule::copy() const {
  // map of stages.
  const HybridScheduleNode* self = operator->();
  std::unordered_map<Stage, Stage, ObjectPtrHash, ObjectPtrEqual> smap;
  ObjectPtr<HybridScheduleNode> n = make_object<HybridScheduleNode>();
  n->outputs = self->outputs;
  // Copy the stages.
  for (Stage s : self->stages) {
    Stage scopy = CopyStage(s);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }
  for (Stage g : self->groups) {
    Stage gcopy = CopyStage(g);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }
  // Remaps the reference relations.
  for (auto kv : self->stage_map) {
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }
  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return HybridSchedule(n);
}

Stage HybridSchedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  ICHECK(it != (*this)->stage_map.end())
      << "Cannot find Stage for operator " << op << " in the hybridschedule";
  return (*it).second;
}

Stage LeastCommonAncestor(Stage g1, Stage g2) {
  if (!g1.defined()) return g1;
  if (!g2.defined()) return g2;
  if (g1.same_as(g2)) return g1;
  Stage g = g1;
  while (g.defined()) {
    if (g.same_as(g2)) return g2;
    g = g->group;
  }
  g = g2;
  while (g.defined()) {
    if (g.same_as(g1)) return g1;
    g = g->group;
  }
  return g;
}

Array<Tensor> RemapTensor(HybridScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      ICHECK(self->stage_map.count(t->op)) << "Given tensor is not in the hybrid_schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the hybrid_schedule stages.
Stage HybridSchedule::create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs) {
  HybridScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops =
      hybrid::GetSubGraph(RemapTensor(self, outputs), RemapTensor(self, inputs), include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<Stage, Entry, ObjectPtrHash, ObjectPtrEqual> counter;
  // The parent group;
  Stage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    Stage op_group = it->second->group;
    if (i == 0) {
      parent_group = op_group;
    } else {
      parent_group = LeastCommonAncestor(parent_group, op_group);
    }
    if (op_group.defined()) {
      ++counter[op_group].count;
    }
  }
  // Create the new group stage.
  Stage gstage(make_object<StageNode>());
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<Stage> stack;
  for (auto& kv : counter) {
    if (!kv.first.same_as(parent_group)) {
      if (kv.first->num_child_stages == kv.second.count) {
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    Stage g = stack.back();
    stack.pop_back();
    if (g->group.defined() && !g->group.same_as(parent_group)) {
      Entry& e = counter[g->group];
      ++e.count;
      if (e.count == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  // Verification and remappig the subgroups.
  for (auto& kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    ICHECK_EQ(kv.first->num_child_stages, kv.second.count)
        << "Trying to group region that intersect with an already existed group";
    if (kv.first->group.same_as(parent_group)) {
      Stage s = kv.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Remap the group of op stages.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->group.same_as(parent_group)) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Correct the attach to keep everything in group.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->attach_type == kScope) {
      Stage cg = LeastCommonAncestor(s->attach_stage->group, gstage);
      if (!cg.same_as(gstage)) {
        LOG(WARNING) << "group invalidates some previous compute_at relation "
                     << " and keeps things to be computed inside the group";
        s.compute_root();
      }
    }
  }

  self->groups.push_back(gstage);
  return gstage;
}

void HybridScheduleNode::InvalidateCache() { op2stage_cache_.clear(); }

void HybridScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (Stage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  ICHECK_EQ(op2stage_cache_.size(), stages.size());
}

bool HybridScheduleNode::Contain(const Operation& op) const {
  return stage_map.find(op) != stage_map.end();
}

HybridSchedule::HybridSchedule(Array<Operation> ops) {
  auto n = make_object<HybridScheduleNode>();
  data_ = n;
  n->outputs = ops;
  auto g = hybrid::CreateReadGraph(n->outputs);
  Array<Operation> post_order = hybrid::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op);
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    // mark scan updates.
    if (const ScanOpNode* scan = op.as<ScanOpNode>()) {
      Array<Tensor> inputs;
      for (Tensor t : scan->state_placeholder) {
        inputs.push_back(t);
      }
      for (Tensor t : scan->inputs) {
        inputs.push_back(t);
      }
      // Create the scan group.
      Stage scan_group = this->create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        Stage s = n->stage_map[scan->update[i]->op];
        ICHECK(scan_group.same_as(s->group));
      }
    }
  }
}

TVM_REGISTER_NODE_TYPE(HybridScheduleNode);

TVM_REGISTER_GLOBAL("ditto.CreateHybridSchedule").set_body_typed(create_hybrid_schedule);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleNormalize").set_body_method(&HybridSchedule::normalize);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCreateGroup").set_body_method(&HybridSchedule::create_group);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCacheRead").set_body_method(&HybridSchedule::cache_read);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCacheWrite").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[1].IsObjectRef<Tensor>()) {
    *ret = args[0].operator HybridSchedule().cache_write(args[1].operator Tensor(), args[2]);
  } else {
    *ret = args[0].operator HybridSchedule().cache_write(args[1].operator Array<Tensor>(), args[2]);
  }
});

TVM_REGISTER_GLOBAL("ditto.HybridScheduleRFactor").set_body_method(&HybridSchedule::rfactor);

// TVM_REGISTER_GLOBAL("ditto.HybridScheduleSlice")
//     .set_body_typed([](HybridSchedule rsch, Tensor tensor, IterVar axis, PrimExpr slice_point) {
//       Tensor tensor_l, tensor_r;
//       rsch.slice(tensor, axis, slice_point, &tensor_l, &tensor_r);
//       return Array<Tensor>({tensor_l, tensor_r});
//     });

}  // namespace hybrid
}  // namespace ditto
