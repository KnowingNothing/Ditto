#include <unordered_set>
#include <vector>

#include <auto_compute/patterns/utils.h>
#include <auto_tensorize/analysis.h>
#include <auto_tensorize/pattern.h>

using namespace tvm;
namespace ditto {
using namespace ditto::auto_compute;
namespace auto_tensorize {

bool IsCubic(te::Operation op, int substantial) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->body.size() != 1U) {
    return false;
  }
  int num_inputs = (int)cop->InputTensors().size();
  if (num_inputs < 2) {
    return false;
  }
  Array<te::Tensor> inputs = cop->InputTensors();

  std::function<int(tir::IterVar iv)> helper;
  helper = [&](tir::IterVar iv) {
    for (int i = 0; i < num_inputs; ++i) {
      te::Operation cur_op = inputs[i]->op;
      IndexIn pi1(cur_op, iv->var);
      if (pi1.check_exist(cop->body[0])) {
        bool is_iv_outer = true;
        for (int j = 0; j < num_inputs; ++j) {
          if (i == j) {
            continue;
          }
          te::Operation other_op = inputs[j]->op;
          IndexIn pi2(other_op, iv->var);
          if (pi2.check_exist(cop->body[0])) {
            is_iv_outer = false;
            break;
          }
        }
        if (is_iv_outer) {
          return i;
        }
      }
    }

    return -1;
  };
  // iv is in outer_ivs iff it appears in non-reduction axes and belongs to single input tensor
  std::vector<tir::IterVar> outer_ivs;
  // the input tensor id that has a outer_iv
  std::unordered_set<int> outer_input_pos;
  for (auto iv : cop->axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (as_int && as_int->value >= substantial) {
      int pos = helper(iv);
      if (pos >= 0) {
        outer_ivs.push_back(iv);
        outer_input_pos.insert(pos);
      }
    }
  }
  bool is_outer = true;
  for (int pos = 0; pos < num_inputs; ++pos) {
    if (!outer_input_pos.count(pos)) {
      is_outer = false;
      break;
    }
  }
  // iv is in inner_ivs iff iv is an reduce_axis and it appears in all input tensors
  std::vector<tir::IterVar> inner_ivs;
  for (auto iv : cop->reduce_axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (as_int && as_int->value >= substantial) {
      bool is_iv_inner = true;
      for (auto inp : inputs) {
        IndexIn pi(inp->op, iv->var);
        if (!pi.check_exist(cop->body[0])) {
          is_iv_inner = false;
          break;
        }
      }
      if (is_iv_inner) {
        inner_ivs.push_back(iv);
      }
    }
  }
  bool is_inner = (inner_ivs.size() > 0U);

  return (is_outer && is_inner) ||
         (op->tag.find("PATTERN_CUBIC") != std::string::npos);
}

bool IsAllred(te::Operation op, int substantial) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->body.size() != 1U) {
    return false;
  }
  int num_inputs = (int)cop->InputTensors().size();
  if (num_inputs != 1) {
    return false;
  }
  Array<te::Tensor> inputs = cop->InputTensors();

  std::vector<tir::IterVar> inner_ivs;
  for (auto iv : cop->reduce_axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (as_int && as_int->value >= substantial) {
      bool is_iv_inner = true;
      for (auto inp : inputs) {
        IndexIn pi(inp->op, iv->var);
        if (!pi.check_exist(cop->body[0])) {
          is_iv_inner = false;
          break;
        }
      }
      if (is_iv_inner) {
        inner_ivs.push_back(iv);
      }
    }
  }
  bool is_inner = (inner_ivs.size() > 0U);
  return is_inner || (op->tag.find("PATTERN_ALLRED") != std::string::npos);
}

bool IsShuffle(te::Operation op) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->body.size() != 1U) {
    return false;
  }
  Map<tir::Var, PrimExpr> vmap;
  for (auto iv : cop->reduce_axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (!as_int || as_int->value > 1) {
      return false;
    } else {
      vmap.Set(iv->var, 0);
    }
  }
  int num_inputs = (int)cop->InputTensors().size();
  if (num_inputs != 1) {
    return false;
  }

  Array<PrimExpr> indices;
  for (auto iv : cop->axis) {
    indices.push_back(iv->var);
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (as_int && as_int->value == 1) {
      vmap.Set(iv->var, 0);
    }
  }

  Array<te::Tensor> inputs = cop->InputTensors();
  arith::Analyzer ana;
  FlattenIndices fi(inputs[0]->op);
  Array<PrimExpr> flatten = fi.flatten(cop->body[0]);
  Array<PrimExpr> simple_flatten;
  for (auto expr : flatten) {
    simple_flatten.push_back(ana.Simplify(tir::Substitute(expr, vmap)));
  }

  PrimExpr target_flatten = flatten_indices(op.output(0)->shape, indices);
  target_flatten = ana.Simplify(tir::Substitute(target_flatten, vmap));

  bool shuffle = false;
  for (auto expr : simple_flatten) {
    PrimExpr res = ana.Simplify(target_flatten - expr);
    const IntImmNode *as_int = res.as<IntImmNode>();
    if (!as_int || as_int->value != 0) {
      shuffle = true;
      break;
    }
  }

  return shuffle || (op->tag.find("PATTERN_SHUFFLE") != std::string::npos);
}

bool IsLocal(te::Operation op, int substantial) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->body.size() != 1U) {
    return false;
  }

  Map<tir::Var, PrimExpr> vmap;
  for (auto iv : cop->reduce_axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (!as_int || as_int->value >= substantial) {
      return false;
    }
  }
  for (auto iv : cop->axis) {
    vmap.Set(iv->var, iv->var + 1);
  }

  Array<te::Tensor> inputs = cop->InputTensors();
  arith::Analyzer ana;
  bool local = true;
  for (auto inp : inputs) {
    FlattenIndices fi(inp->op);
    Array<PrimExpr> flatten = fi.flatten(cop->body[0]);
    Array<PrimExpr> simple_flatten;
    for (auto expr : flatten) {
      simple_flatten.push_back(ana.Simplify(expr));
    }

    PrimExpr new_body = tir::Substitute(cop->body[0], vmap);
    Array<PrimExpr> delta_flatten = fi.flatten(new_body);
    Array<PrimExpr> simple_delta_flatten;
    for (auto expr : delta_flatten) {
      simple_delta_flatten.push_back(ana.Simplify(expr));
    }

    int num_expr = (int)flatten.size();
    CHECK((int)delta_flatten.size() == num_expr);
    for (int i = 0; i < num_expr; ++i) {
      PrimExpr res = ana.Simplify(simple_delta_flatten[i] - simple_flatten[i]);
      const tir::IntImmNode *as_int = res.as<tir::IntImmNode>();
      if (!as_int) {
        local = false;
        break;
      }
    }

    if (!local) {
      break;
    }
  }

  return local || (op->tag.find("PATTERN_LOCAL") != std::string::npos);
}

bool IsView(te::Operation op) {
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (!cop) {
    return false;
  }
  if (cop->body.size() != 1U) {
    return false;
  }
  const tir::ProducerLoadNode *as_load =
      cop->body[0].as<tir::ProducerLoadNode>();
  if (!as_load) {
    return false;
  }
  Map<tir::Var, PrimExpr> vmap;
  for (auto iv : cop->reduce_axis) {
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (!as_int || as_int->value > 1) {
      return false;
    } else {
      vmap.Set(iv->var, 0);
    }
  }
  int num_inputs = (int)cop->InputTensors().size();
  if (num_inputs != 1) {
    return false;
  }

  Array<PrimExpr> indices;
  for (auto iv : cop->axis) {
    indices.push_back(iv->var);
    const tir::IntImmNode *as_int = iv->dom->extent.as<tir::IntImmNode>();
    if (as_int && as_int->value == 1) {
      vmap.Set(iv->var, 0);
    }
  }

  Array<te::Tensor> inputs = cop->InputTensors();
  arith::Analyzer ana;
  FlattenIndices fi(inputs[0]->op);
  Array<PrimExpr> flatten = fi.flatten(cop->body[0]);
  Array<PrimExpr> simple_flatten;
  for (auto expr : flatten) {
    simple_flatten.push_back(ana.Simplify(tir::Substitute(expr, vmap)));
  }

  PrimExpr target_flatten = flatten_indices(op.output(0)->shape, indices);
  target_flatten = ana.Simplify(tir::Substitute(target_flatten, vmap));

  bool view = true;
  for (auto expr : simple_flatten) {
    PrimExpr res = ana.Simplify(target_flatten - expr);
    const IntImmNode *as_int = res.as<IntImmNode>();
    if (!as_int || as_int->value != 0) {
      view = false;
      break;
    }
  }

  return view || (op->tag.find("PATTERN_VIEW") != std::string::npos);
}

OpPattern GetOpPattern(te::Operation op) {
  if (IsCubic(op)) {
    return OpPattern::PATTERN_CUBIC;
  } else if (IsAllred(op)) {
    return OpPattern::PATTERN_ALLRED;
  } else if (IsShuffle(op)) {
    return OpPattern::PATTERN_SHUFFLE;
  } else if (IsLocal(op)) {
    return OpPattern::PATTERN_LOCAL;
  } else if (IsView(op)) {
    return OpPattern::PATTERN_VIEW;
  } else {
    LOG(WARNING) << "Can't judge the pattern of op";
    return OpPattern::PATTERN_CUBIC; 
  }
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.IsCubic").set_body_typed(IsCubic);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.IsAllred").set_body_typed(IsAllred);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.IsShuffle").set_body_typed(IsShuffle);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.IsLocal").set_body_typed(IsLocal);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.IsView").set_body_typed(IsView);
TVM_REGISTER_GLOBAL("ditto.auto_tensorize.GetOpPattern")
    .set_body_typed([](te::Operation op) {
      switch (GetOpPattern(op)) {
      case OpPattern::PATTERN_CUBIC:
        return "PATTERN_CUBIC";
        break;

      case OpPattern::PATTERN_ALLRED:
        return "PATTERN_ALLRED";
        break;

      case OpPattern::PATTERN_SHUFFLE:
        return "PATTERN_SHUFFLE";
        break;

      case OpPattern::PATTERN_LOCAL:
        return "PATTERN_LOCAL";
        break;

      case OpPattern::PATTERN_VIEW:
        return "PATTERN_VIEW";
        break;

      default:
        CHECK(false) << "Can't judge the pattern of op:\n" << op << "\n";
        break;
      }
    });

} // namespace auto_tensorize

} // namespace ditto