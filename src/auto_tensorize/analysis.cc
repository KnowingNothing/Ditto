#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <auto_tensorize/analysis.h>

using namespace tvm;
namespace ditto {

namespace auto_tensorize {

PrimExpr flatten_indices(const Array<PrimExpr> shape,
                         const Array<PrimExpr> indices) {
  int num_dim = (int)shape.size();
  CHECK((int)indices.size() == num_dim && num_dim > 0);
  PrimExpr flatten = indices[num_dim - 1];
  for (int i = 0; i < num_dim - 1; ++i) {
    flatten = indices[num_dim - i - 2] * shape[num_dim - i - 1] + flatten;
  }
  return flatten;
}



Array<Array<tir::IterVar>> share_axis_analysis(te::Operation op1,
                                               te::Operation op2) {
  // consumer_op -> axis -> producer_op -> set(axis)
  std::unordered_map<
      te::Operation,
      std::unordered_map<
          tir::IterVar,
          std::unordered_map<te::Operation, std::unordered_set<tir::IterVar>>>>
      share_relations;

  std::unordered_set<te::Operation> visit;
  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (visit.count(op)) {
      return;
    }
    const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
    if (!cop) {
      return;
    }
    CHECK(cop->body.size() == 1U);

    for (auto inp : cop->InputTensors()) {
      helper(inp->op);
    }

    // step 1 relations
    for (auto siv : cop->axis) {
      std::unordered_map<te::Operation, std::unordered_set<tir::IterVar>> tmp;
      for (auto inp : cop->InputTensors()) {
        const te::ComputeOpNode *inp_cop = inp->op.as<te::ComputeOpNode>();
        if (!inp_cop)
          continue;
        int num_input_dim = (int)inp->shape.size();
        for (int i = 0; i < num_input_dim; ++i) {
          IndexInDim iid(inp->op, siv->var, i);
          if (iid.check_exist(cop->body[0])) {
            tmp[inp->op].insert(inp_cop->axis[i]);
          }
        }
      }
      if (tmp.size()) {
        share_relations[op][siv] = tmp;
      }
    }

    for (auto riv : cop->reduce_axis) {
      std::unordered_map<te::Operation, std::unordered_set<tir::IterVar>> tmp;
      for (auto inp : cop->InputTensors()) {
        const te::ComputeOpNode *inp_cop = inp->op.as<te::ComputeOpNode>();
        if (!inp_cop)
          continue;
        int num_input_dim = (int)inp->shape.size();
        for (int i = 0; i < num_input_dim; ++i) {
          IndexInDim iid(inp->op, riv->var, i);
          if (iid.check_exist(cop->body[0])) {
            tmp[inp->op].insert(inp_cop->axis[i]);
          }
        }
      }
      if (tmp.size()) {
        share_relations[op][riv] = tmp;
      }
    }

    // global relations through join
    // if a -> iv1 -> b -> iv2
    // and b -> iv3 -> c -> iv4
    // then update a -> iv1 -> c -> iv4
    std::unordered_map<
        tir::IterVar,
        std::unordered_map<te::Operation, std::unordered_set<tir::IterVar>>>
        update;
    if (share_relations.count(op)) {
      for (auto iv_map1 : share_relations.at(op)) {
        for (auto op_vec : iv_map1.second) {
          if (share_relations.count(op_vec.first)) {
            for (auto iv_map2 : share_relations.at(op_vec.first)) {
              for (auto rel_iv : op_vec.second) {
                if (rel_iv == iv_map2.first) {
                  update[iv_map1.first] = iv_map2.second;
                }
              }
            }
          }
        }
      }
    }
    for (auto kv : update) {
      for (auto kkvv : kv.second) {
        for (auto v : kkvv.second) {
          share_relations[op][kv.first][kkvv.first].insert(v);
        }
      }
    }
  };

  helper(op2);
  Array<Array<tir::IterVar>> ret;
  if (share_relations.count(op2)) {
    for (auto kv : share_relations.at(op2)) {
      for (auto kkvv : kv.second) {
        if (kkvv.first == op1) {
          for (auto iv : kkvv.second) {
            Array<tir::IterVar> tmp;
            tmp.push_back(iv);
            tmp.push_back(kv.first);
            if (tmp.size()) {
              ret.push_back(tmp);
            }
          }
        }
      }
    }
  }

  return ret;
}
TVM_REGISTER_NODE_TYPE(FeatureLogNode);

FeatureLog::FeatureLog(
                      Map<tir::Var, IntImm> bounds,\
                      int op1MemVisit,\
                      int op1WorkLoad,\
                      int op1Buffer,\
                      int op2MemVisit,\
                      int op2WorkLoad,\
                      int op2Buffer,\
                      int parallelism,\
                      hardware::HardwareParam hp
                      ){
  auto n = make_object<FeatureLogNode>();
  n->bounds = bounds;
  n->op1.memVisit = op1MemVisit;
  n->op1.workLoad = op1WorkLoad;
  n->op1.bufferSize = op1Buffer;
  n->op2.memVisit = op2MemVisit;
  n->op2.workLoad = op2WorkLoad;
  n->op2.bufferSize = op2Buffer;
  n->parallelism = parallelism;
  n->hardwareParam = hp;
  data_ = n;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<FeatureLogNode>([](const ObjectRef& node, ReprPrinter* p) {
            auto* op = static_cast<const FeatureLogNode*>(node.get());
            p->PrintIndent();
            p->stream << "Features(\n";
            p->stream << "bounds: ";
            p->Print(op->bounds);
            p->stream << ", ";
            p->stream << "op1MemVisit: " << op->op1.memVisit << ", ";
            p->stream << "op1WorkLoad: " << op->op1.workLoad << ", ";
            p->stream << "op1BufferSize: " << op->op1.bufferSize << ",\n"; 
            p->stream << "op2MemVisit: " << op->op2.memVisit << ", ";
            p->stream << "op2WorkLoad: " << op->op2.workLoad << ", ";
            p->stream << "op2BufferSize: " << op->op2.bufferSize << ",\n"; 
            p->stream << "parallelism: " << op->parallelism << ")\n";
        });

FeatureLog buildFeatureLog(IterGraph ig, hardware::HardwareParam hp){
  return FeatureLog(
    ig->inferBound(),\
    ig->getFirstOpDataVolume(),\
    ig->getFirstOpWorkload(),\
    ig->getFirstOpBufferSize(),\
    ig->getSecondOpDataVolume(),\
    ig->getSecondOpWorkload(),\
    ig->getSecondOpBufferSize(),\
    ig->getParallelism(),\
    hp
  );
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.ShareAxisAnalysis")
    .set_body_typed(share_axis_analysis);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildFeatureLog")
    .set_body_typed(buildFeatureLog);
} // namespace auto_tensorize

} // namespace ditto