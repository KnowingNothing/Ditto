#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <vector>
#include <memory>
#include <string>

#include <autograd/autodiff/autodiff.h>
#include <utils/logging.h>

namespace ditto {
using namespace utils;
namespace autograd {

PrimExpr ExprReMapper::VisitExpr_(const VarNode *op) {
  if (var_map.find(op) != var_map.end()) {
    return var_map[op];
  }
  Var ret = Var(get_new_var_name(), op->dtype);
  var_map[op] = ret;
  return ret;
}

PrimExpr ExprReMapper::VisitExpr_(const SizeVarNode *op) {
  if (size_var_map.find(op) != size_var_map.end()) {
    return size_var_map[op];
  }
  SizeVar ret = SizeVar(get_new_var_name(), op->dtype);
  size_var_map[op] = ret;
  return ret;
}

PrimExpr ExprReMapper::VisitExpr_(const ProducerLoadNode *op) {
  Array<PrimExpr> new_args;
  for (auto v : op->indices) {
    new_args.push_back(VisitExpr(v));
  }

  if (call_map.find(op->producer) != call_map.end()) {
    return ProducerLoad(call_map[op->producer], op->indices);
  } else {
    te::Tensor new_tensor = get_new_tensor(Downcast<te::Tensor>(op->producer));
    call_map[op->producer] = new_tensor;
    return ProducerLoad(new_tensor, op->indices);
  }
}

PrimExpr ExprReMapper::VisitExpr_(const ReduceNode *op) {
  CommReducer reducer;
  Array<Var> lhs;
  Array<Var> rhs;
  Array<PrimExpr> results;
  Array<PrimExpr> identities;
  for (Var l : op->combiner->lhs) {
    if (var_map.find(l.get()) != var_map.end()) {
      lhs.push_back(var_map[l.get()]);
    } else {
      VisitExpr(l);
      lhs.push_back(var_map[l.get()]);
    }
  }
  for (auto r : op->combiner->rhs) {
    if (var_map.find(r.get()) != var_map.end()) {
      rhs.push_back(var_map[r.get()]);
    } else {
      VisitExpr(r);
      rhs.push_back(var_map[r.get()]);
    }
  }
  for (auto r : op->combiner->result) {
    results.push_back(VisitExpr(r));
  }
  for (auto i : op->combiner->identity_element) {
    identities.push_back(VisitExpr(i));
  }
  reducer = CommReducer(lhs, rhs, results, identities);

  Array<PrimExpr> source;
  for (auto s : op->source) {
    source.push_back(VisitExpr(s));
  }

  Array<IterVar> axis;
  for (auto iv : op->axis) {
    VisitExpr(iv->var);
    axis.push_back(IterVar(iv->dom, var_map[iv->var.get()], iv->iter_type,
                           iv->thread_tag));
  }

  PrimExpr condition = this->VisitExpr(op->condition);

  return Reduce(reducer, source, axis, condition, op->value_index, op->init);
}

std::string generate_tag_from_body(Array<IterVar> axis, Array<PrimExpr> body) {
  std::ostringstream oss;
  oss.str("");
  if (body.size() == 0U) {
    ERROR << "Unexpected empty body!";
  }

  const ReduceNode *as_reduce = body[0].as<ReduceNode>();

  if (as_reduce != nullptr) {
    CHECK(body.size() == 1U) << "Only support reduce with one body.";
    Array<IterVar> axis_;
    for (auto iv : axis) {
      axis_.push_back(iv);
    }
    for (auto iv : as_reduce->axis) {
      axis_.push_back(iv);
    }
    ExprReMapper remapper(axis_);
    PrimExpr new_reduce = remapper(body[0]);
    const ReduceNode *as_reduce = new_reduce.as<ReduceNode>();
    CHECK(as_reduce != nullptr);

    oss << "R[";
    bool add_colon = false;
    for (auto s : axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << s->dom->extent;
    }
    oss << "] [";
    add_colon = false;
    for (auto iv : as_reduce->axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << iv->dom->extent;
    }
    oss << "] { ";
    oss << as_reduce->combiner;
    oss << " } { ";
    for (size_t i = 0; i < as_reduce->source.size(); ++i) {
      if (i != 0) {
        oss << "; ";
      }
      oss << as_reduce->source[i];
    }
    oss << " }";
  } else {
    // not reduce
    oss << "S[";
    bool add_colon = false;
    for (auto s : axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << s->dom->extent;
    }
    oss << "] [ ] { } { ";
    bool add_semicolon = false;
    for (auto b : body) {
      CHECK(b.as<ReduceNode>() == nullptr)
          << "Should only contain non-reduce expr.";
      ExprReMapper remapper(axis);
      PrimExpr new_b = remapper(b);
      if (add_semicolon) {
        oss << "; ";
      } else {
        add_semicolon = true;
      }
      oss << new_b;
    }
    oss << " }";
  }

  return oss.str();
}

namespace {

Tensor ones_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices) {
    return make_const(tensor->dtype, 1);
  };
  Array<IterVar> axis;
  for (auto s : shape) {
    axis.push_back(IterVar(Range(0, s), Var(""), IterVarType::kDataPar));
  }
  std::string tag =
      generate_tag_from_body(axis, {make_const(tensor->dtype, 1)});
  return te::compute(shape, func, "ones_" + tensor->op->name, tag, {});
}

Tensor zeros_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices) {
    return make_const(tensor->dtype, 0);
  };

  Array<IterVar> axis;
  Array<Var> vars;
  for (auto s : shape) {
    auto var = Var("");
    vars.push_back(var);
    axis.push_back(IterVar(Range(0, s), var, IterVarType::kDataPar));
  }
  std::string tag = generate_tag_from_body(axis, {func(vars)});
  return te::compute(shape, func, "zeros_" + tensor->op->name, tag, {});
}

Tensor grad_intra_op(const Tensor &input, const Tensor &output,
                     const Tensor &doutput) {
  return grad_op(input, output, doutput);
}

Tensor collect_rule(const Tensor &input, const Array<Tensor> &outputs,
                    const Array<Tensor> &grad_outputs) {
  CHECK(outputs.size() > 0)
      << "No effective gradients from outputs, did you forget to set "
         "`requires_grad=True` for consumers of "
      << input << "?\n";
  CHECK(outputs.size() == grad_outputs.size()) << "Length mismatch.\n";
  Array<Tensor> partial_grads;
  size_t num_outputs = outputs.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    partial_grads.push_back(grad_intra_op(input, outputs[i], grad_outputs[i]));
  }
  if (num_outputs == 1U) {
    return partial_grads[0];
  }
  Array<PrimExpr> shape = input->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&partial_grads, &num_outputs](const Array<Var> &input_indices) {
    // num_outputs should > 0, because otherwise, this function won't be used
    PrimExpr res = partial_grads[0](input_indices);
    for (size_t i = 1; i < num_outputs; ++i) {
      res = Add(res, partial_grads[i](input_indices));
    }
    return res;
  };
  // std::string dim = std::to_string(input->shape.size());
  // std::string num = std::to_string(num_outputs);
  Array<Var> indices;
  Array<IterVar> axis;
  for (auto s : shape) {
    auto var = Var("");
    indices.push_back(var);
    axis.push_back(IterVar(Range(0, s), var, IterVarType::kDataPar));
  }
  PrimExpr res = partial_grads[0](indices);
  for (size_t i = 1; i < num_outputs; ++i) {
    res = Add(res, partial_grads[i](indices));
  }
  std::string tag = generate_tag_from_body(axis, {res});
  return te::compute(shape, func, "collect_" + input->op->name, tag, {});
}

} // anonymous namespace

Array<Tensor> Gradient(const Tensor &output, const Array<Tensor> &weights,
                       const Tensor &doutput_or_null) {
  Tensor doutput = doutput_or_null.get() ? doutput_or_null : ones_like(output);

  std::unordered_map<Tensor, Array<Tensor>> feed_graph;
  std::vector<Tensor> stack;
  stack.push_back(output);

  while (!stack.empty()) {
    Tensor current = stack.back();
    stack.pop_back();
    for (const Tensor &input : current->op->InputTensors()) {
      if (!feed_graph.count(input)) {
        stack.push_back(input);
      }
      feed_graph[input].push_back(current);
    }
  }

  std::unordered_map<Tensor, Tensor> grad_map;

  grad_map[output] = doutput;

  std::function<Tensor(const Tensor &)> get_grad_compute;
  get_grad_compute = [&get_grad_compute, &grad_map,
                      &feed_graph](const Tensor &tensor) {
    if (!grad_map.count(tensor)) {
      // Here the gradient hasn't been computed yet
      Tensor tensor_grad;
      // if (!tensor->requires_grad) {
      //   LOG(WARNING) << "grad to tensor that doesn't requires grad: " << tensor
      //                << ".\n";
      //   tensor_grad = zeros_like(tensor);
      //   grad_map[tensor] = tensor_grad;
      //   return tensor_grad;
      // }
      // Need to compute gradients
      Array<Tensor> direct_consumers = feed_graph[tensor];
      if (direct_consumers.empty()) {
        LOG(WARNING) << "grad to tensor that doesn't have consumers.\n";
        tensor_grad = zeros_like(tensor);
      } else {
        Array<Tensor> grad_outputs;
        Array<Tensor> effective_consumers;
        for (const Tensor &direct_consumer : direct_consumers) {
          // if (direct_consumer->requires_grad) {
          effective_consumers.push_back(direct_consumer);
          grad_outputs.push_back(get_grad_compute(direct_consumer));
          // }
        }
        tensor_grad = collect_rule(tensor, effective_consumers, grad_outputs);
      }

      grad_map[tensor] = tensor_grad;
      return tensor_grad;
    } else {
      return grad_map[tensor];
    }
  };

  Array<Tensor> result;
  for (const Tensor &weight : weights) {
    result.push_back(get_grad_compute(weight));
  }
  return result;
}

TVM_REGISTER_GLOBAL("ditto.autograd.generate_tag_from_body")
    .set_body_typed([](Array<IterVar> axis, Array<PrimExpr> body) {
      return generate_tag_from_body(axis, body);
    });

TVM_REGISTER_GLOBAL("ditto.autograd.Gradient").set_body([](TVMArgs args, TVMRetValue *ret) {
  // LOG(WARNING) << "tg.Gradient is an experimental feature.";
  if (args.size() == 2) {
    *ret = Gradient(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = Gradient(args[0], args[1], args[2]);
  }
});

} // namespace autograd
} // namespace ditto
