#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <auto_tensorize/dse/evaluator.h>
using namespace tvm;
namespace ditto {
namespace auto_tensorize {
TVM_REGISTER_NODE_TYPE(FusionResultNode);
FusionResult::FusionResult(Map<tir::Var, IntImm> bounds, double op1MemVisit,
                           double op1WorkLoad, double op1Buffer,
                           double op2MemVisit, double op2WorkLoad,
                           double op2Buffer, double locality,
                           double parallelism, double redundancy,
                           double n_block, bool valid, int fusionLevel,
                           int bytePerEle, double cacheSize, double memUse) {
  auto n = make_object<FusionResultNode>();
  n->bounds = bounds;
  n->op1.dataMovementVolume = op1MemVisit;
  n->op1.workLoad = op1WorkLoad;
  n->op1.bufferSize = op1Buffer;
  n->op2.dataMovementVolume = op2MemVisit;
  n->op2.workLoad = op2WorkLoad;
  n->op2.bufferSize = op2Buffer;
  n->locality = locality;
  n->parallelism = parallelism;
  n->redundancy = redundancy;
  n->n_block = n_block;
  n->valid = valid;
  n->fusionLevel = fusionLevel;
  n->bytePerEle = bytePerEle;
  n->workload = op1WorkLoad + op2WorkLoad;
  n->cacheSize = cacheSize;
  n->occupancy = std::max(op1Buffer, op2Buffer) / cacheSize;
  n->dataMovement = (op1MemVisit + op2MemVisit) / (double)parallelism;
  n->memUse = memUse;
  data_ = n;
}
double FusionResultNode::getArithmeticIntensity() const {
  // the arithmetic intensity;
  return (op1.workLoad + op2.workLoad) /
         double(op1.dataMovementVolume + op2.dataMovementVolume);
}
double FusionResultNode::getWorkload() const {
  return op1.workLoad + op2.workLoad;
}
StaticAnalysis::StaticAnalysis(IterGraph ig) {
  auto n = make_object<StaticAnalysisNode>();
  n->tag = "static analysis";
  n->iterGraph = ig;
  data_ = n;
}

Result StaticAnalysisNode::eval(Item it) const {
  auto fusionItem = Downcast<FusionItem, Item>(it);
  iterGraph->setFusion(fusionItem);
  FusionResult result = iterGraph->getAnalyticalResult();
  return result;
}
cost_t StaticAnalysisNode::cost(Item it) const {
  auto fusionItem = Downcast<FusionItem, Item>(it);
  CHECK(it.defined());
  iterGraph->setFusion(fusionItem);
  FusionResult result = iterGraph->getAnalyticalResult();
  if (!result->valid)
    return INFINITY;
  double dm =
      (result->op1.dataMovementVolume + result->op2.dataMovementVolume) /
      (double)result->parallelism;
  return dm;
}
Map<String, FloatImm> FusionResultNode::getLog() const {
  Map<String, FloatImm> m;
  m.Set("locality", FloatImm(DataType::Float(64), locality));
  m.Set("redundancy", FloatImm(DataType::Float(64), (double)redundancy));
  m.Set("parallelism", FloatImm(DataType::Float(64), (double)parallelism));
  m.Set("AI", FloatImm(DataType::Float(64), getArithmeticIntensity()));
  m.Set("valid", FloatImm(DataType::Float(64), (double)valid));
  return m;
}
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EvaluatorNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const EvaluatorNode *>(node.get());
      p->stream << op->tag;
    });
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FusionResultNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const FusionResultNode *>(node.get());
      p->stream << "FusionResult(\n";
      p->stream << "op1:\n";
      p->stream << "\tDataVolume: " << op->op1.dataMovementVolume << "B\n";
      p->stream << "\tworkLoad: " << op->op1.workLoad << "flops\n";
      p->stream << "\tbufferSize: " << op->op1.bufferSize << "B\n";
      p->stream << "op2:\n";
      p->stream << "\tDataVolume: " << op->op2.dataMovementVolume << "B\n";
      p->stream << "\tworkLoad: " << op->op2.workLoad << "flops\n";
      p->stream << "\tbufferSize: " << op->op2.bufferSize << "B\n";
      p->stream << "n_block: " << op->n_block << "\n";
      p->stream << "locality: " << op->locality << "\n";
      p->stream << "redundancy: " << op->redundancy << "\n";
      p->stream << "AI: " << op->getArithmeticIntensity() << "\n";
      p->stream << "parallelism: " << op->parallelism << "\n";
      p->stream << "fusionLevel: " << op->fusionLevel << "\n";
      p->stream << "occupancy: " << op->occupancy << "\n";
      p->stream << "cacheSize: " << op->cacheSize << ")";
    });
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getLog")
    .set_body_method<FusionResult>(&FusionResultNode::getLog);
} // namespace auto_tensorize
} // namespace ditto
