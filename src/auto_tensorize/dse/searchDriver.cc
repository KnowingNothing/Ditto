#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <auto_tensorize/dse/searchDriver.h>
using namespace tvm;
namespace ditto {
namespace auto_tensorize {
SearchDriver::SearchDriver(Array<Evaluator> evals, SearchSpace searchSpace,
                           SearchAlg alg) {
  auto n = make_object<SearchDriverNode>();
  n->evals = evals;
  n->alg = alg;
  n->searchSpace = searchSpace;
  data_ = std::move(n);
}
Item SearchDriverNode::search() {
  result = alg->search();
  return result.second;
}
std::pair<cost_t, Item> SearchDriverNode::search_with_loss() {
  result = alg->search();
  return result;
}
Array<Result> SearchDriverNode::eval(Item it) {
  Array<Result> ret;
  for (auto e : evals)
    ret.push_back(e->eval(it));
  return ret;
}
TVM_REGISTER_NODE_TYPE(ResultNode);
std::pair<cost_t, Item> BruteForceNode::search() const {
  bool has_staticAnalysis = false;
  Evaluator eval;
  for (auto eval_ : evals) {
    if (eval_.as<StaticAnalysisNode>()) {
      eval = eval_;
      has_staticAnalysis = true;
      break;
    }
  }
  CHECK(has_staticAnalysis) << "no staticAnalysis evaluator";
  cost_t best_l = INFINITY;
  long long best_i = -1;
  for (long long i = 0; i < (long long)searchSpace->cardinal; i++) {
    Item item = searchSpace->idxToItem(i);

    cost_t tmp_loss = eval->cost(item);
    if (tmp_loss < best_l) {
      best_l = tmp_loss;
      best_i = i;
    }
  }
  if (best_i < 0) {
    LOG(WARNING) << "no valid candidate in current searchspace";
    return {INFINITY, Item()};
  }
  Item best_item = searchSpace->idxToItem(best_i);
  return {best_l, best_item};
}
BruteForce::BruteForce(SearchSpace searchSpace, Array<Evaluator> evals) {
  auto n = make_object<BruteForceNode>();
  n->name = "brute force";
  n->searchSpace = searchSpace;
  n->evals = evals;
  data_ = std::move(n);
}

SearchDriver buildSearchDriver(IterGraph ig, Array<String> evaltypes,
                                      String searcher,
                                      hardware::HardwareParam hw_param,
                                      String dtype) {
  SearchSpace searchSpace = ig->getSearchSpace();
  Array<Evaluator> evals;
  evals.push_back(StaticAnalysis(ig, hw_param, dtype));
  SearchAlg alg;
  if (searcher == "bruteForce" || searcher == "BruteForce" ||
      searcher == "brute") {
    alg = BruteForce(searchSpace, evals);
  } else {
    CHECK(false) << "searcher undefined, candidates are: bruteForce, ";
  }
  return SearchDriver(evals, searchSpace, alg);
}
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SearchAlgNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const SearchAlgNode *>(node.get());
      p->stream << op->name;
    });
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SearchDriverNode>([](const ObjectRef &node, ReprPrinter *p) {
      auto *op = static_cast<const SearchDriverNode *>(node.get());
      p->stream << "----------------searchDriver----------------";
      p->stream << "\nevals: \t";
      p->Print(op->evals);
      p->stream << "\nseachSpace: \t";
      p->Print(op->searchSpace);
      p->stream << "\nalg: \t";
      p->Print(op->alg);
      p->stream << "\n-------------------------------------------";
    });

TVM_REGISTER_NODE_TYPE(SearchDriverNode);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildSearchDriver")
    .set_body_typed(buildSearchDriver);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.search")
    .set_body_method<SearchDriver>(&SearchDriverNode::search);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.eval")
    .set_body_method<SearchDriver>(&SearchDriverNode::eval);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.getFusionSpace")
    .set_body_method<SearchDriver>(&SearchDriverNode::getFusionSpace);
} // namespace auto_tensorize
} // namespace ditto