#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <auto_tensorize/dse/searchSpace.h>
#include <auto_tensorize/iter_graph.h>
#include <hardware/base/hw_param.h>

using namespace tvm;
namespace ditto {
namespace auto_tensorize {
#define cost_t double

class ResultNode : public Object {
public:
  void VisitAttrs(AttrVisitor *v) {}
  static constexpr const char *_type_key = "ditto.auto_tensorize.Result";
  TVM_DECLARE_BASE_OBJECT_INFO(ResultNode, Object);
};

class Result : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Result, ObjectRef, ResultNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ResultNode);
};

class FusionResultNode : public ResultNode {
public:
  struct opFeat {
    double dataMovementVolume;
    double workLoad;
    double bufferSize;
  };
  Map<tir::Var, IntImm> bounds;
  opFeat op1, op2;
  int workload;
  double parallelism;
  double redundancy;
  double locality;
  double n_block;
  bool valid;
  int fusionLevel;
  int bytePerEle;
  double cacheSize;
  double occupancy;
  double dataMovement;

  double getArithmeticIntensity() const;
  double getLocality(hardware::HardwareParam) const;
  double getRedundancy() const;
  double getParallelism() const;
  double getWorkload() const;
  Map<String, FloatImm> getLog() const;
  static constexpr const char *_type_key = "ditto.auto_tensorize.FusionResult";
  void VisitAttrs(AttrVisitor *v) {
    v->Visit("workload", &workload);
  }
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionResultNode, ResultNode);
};

class FusionResult : public Result {
public:
  TVM_DLL FusionResult(Map<tir::Var, IntImm> bounds, double op1MemVisit,
                       double op1WorkLoad, double op1Buffer, double op2MemVisit,
                       double op2WorkLoad, double op2Buffer, double locality,
                       double parallelism, double redundancy, double n_block,
                       bool valid, int fusionLevel, int bytePerEle, double cacheSize);
  TVM_DEFINE_OBJECT_REF_METHODS(FusionResult, Result, FusionResultNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FusionResultNode);
};

class PerformanceResultNode : public ResultNode {
public:
  TVM_DECLARE_FINAL_OBJECT_INFO(PerformanceResultNode, ResultNode);
};

class PerformanceResult : public Result {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(PerformanceResult, Result,
                                PerformanceResultNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PerformanceResultNode);
};

class EvaluatorNode : public Object {
public:
  std::string tag;
  virtual Result eval(Item it) const { return Result(); }
  virtual cost_t cost(Item it) const { return cost_t(INFINITY); }
  static constexpr const char *_type_key = "ditto.auto_tensorize.Evaluator";
  TVM_DECLARE_BASE_OBJECT_INFO(EvaluatorNode, Object);
};

class Evaluator : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Evaluator, ObjectRef, EvaluatorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluatorNode);
};

class StaticAnalysisNode : public EvaluatorNode {
public:
  IterGraph iterGraph;
  hardware::HardwareParam hw_param;
  int bytePerEle;
  Result eval(Item it) const;
  cost_t cost(Item it) const;
  TVM_DECLARE_FINAL_OBJECT_INFO(StaticAnalysisNode, EvaluatorNode);
};

class StaticAnalysis : public Evaluator {
public:
  TVM_DLL StaticAnalysis(IterGraph ig, hardware::HardwareParam hw_param,
                         String dtype);
  TVM_DEFINE_OBJECT_REF_METHODS(StaticAnalysis, Evaluator, StaticAnalysisNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StaticAnalysisNode);
};

} // namespace auto_tensorize

} // namespace ditto