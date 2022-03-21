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

#include <auto_tensorize/dse/evaluator.h>
#include <auto_tensorize/dse/searchSpace.h>
#include <auto_tensorize/iter_graph.h>
#include <hardware/base/hw_param.h>

using namespace tvm;
namespace ditto {
namespace auto_tensorize {
class SearchAlgNode : public Object {
public:
  std::string name;
  SearchSpace searchSpace;
  Array<Evaluator> evals;
  static constexpr const char *_type_key = "ditto.auto_tensorize.SearchAlg";
  virtual std::pair<cost_t, Item> search() const { return {INFINITY, Item()}; }
  TVM_DECLARE_BASE_OBJECT_INFO(SearchAlgNode, Object);
};

class SearchAlg : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(SearchAlg, ObjectRef, SearchAlgNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SearchAlgNode);
};

class BruteForceNode : public SearchAlgNode {
public:
  std::pair<cost_t, Item> search() const override;
  TVM_DECLARE_BASE_OBJECT_INFO(BruteForceNode, SearchAlgNode);
};
class BruteForce : public SearchAlg {
public:
  TVM_DLL BruteForce(SearchSpace searchSpace, Array<Evaluator> evals);
  TVM_DEFINE_OBJECT_REF_METHODS(BruteForce, SearchAlg, BruteForceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BruteForceNode);
};

class SearchDriverNode : public Object {
public:
  Array<Evaluator> evals;
  SearchSpace searchSpace;
  SearchAlg alg;
  std::pair<cost_t, Item> result;
  /*!
   *  \brief main search function
   * 0. register the searchSpace in alg; register the iterGraph in evals (move
   * to build SearchDriver)
   * 1. use alg to decide on the item; out: item
   * 2. use every evaluator to evaluate the item; out: Array<features>
   * 3. give the result back to the alg
   * 4. decide whether to stop;
   * 5. if not, go to 1.
   */
  Item search();

  std::pair<cost_t, Item> search_with_loss();

  FusionSpace getFusionSpace() {
    return Downcast<FusionSpace, SearchSpace>(searchSpace);
  }
  Array<Result> eval(Item it);
  void VisitAttrs(AttrVisitor *v) {
    v->Visit("evals", &evals);
    v->Visit("searchSpace", &searchSpace);
    v->Visit("searchAlg", &alg);
    v->Visit("result", &result.second);
  }
  static constexpr const char *_type_key = "ditto.auto_tensorize.SearchDriver";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchDriverNode, Object);
};

class SearchDriver : public ObjectRef {
public:
  TVM_DLL SearchDriver(Array<Evaluator> evals, SearchSpace searchSpace,
                       SearchAlg alg);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchDriver, ObjectRef,
                                        SearchDriverNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SearchDriverNode);
};

inline SearchDriver buildSearchDriver(IterGraph ig, Array<String> evaltypes,
                                      String searcher,
                                      hardware::HardwareParam hw_param,
                                      String dtype);
} // namespace auto_tensorize

} // namespace ditto