#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <autograd/autodiff/autodiff.h>
#include <autograd/autograd.h>

#include <vector>

namespace ditto {
using namespace tvm;
namespace autograd {

Array<te::Tensor> combine_results(std::vector<Array<te::Tensor>> all_grads) {
  int length = (int)all_grads.size();
  CHECK(length >= 1) << "It doesn't make sense to combine zero length grads.\n";
  if (length == 1) {
    return all_grads[0];
  }
  int num_grads = (int)all_grads[0].size();
  for (auto grads : all_grads) {
    CHECK((int)grads.size() == num_grads) << "Grads length mismatch.\n";
  }
  Array<te::Tensor> ret;
  for (int l = 0; l < num_grads; ++l) {
    std::function<PrimExpr(const Array<Var> &input_indices)> func;
    func = [&](const Array<tir::Var> &input_indices) {
      PrimExpr res = all_grads[0][l](input_indices);
      for (int i = 1; i < length; ++i) {
        res = tir::Add(res, all_grads[i][l](input_indices));
      }
      return res;
    };
    Array<tir::Var> indices;
    Array<tir::IterVar> axis;
    for (auto s : all_grads[0][l]->shape) {
      auto var = Var("iv");
      indices.push_back(var);
      axis.push_back(IterVar(Range(0, s), var, IterVarType::kDataPar));
    }
    std::string tag = "";
    te::Tensor combined =
        te::compute(all_grads[0][l]->shape, func,
                    "combine_" + all_grads[0][l]->op->name, tag, {});
    ret.push_back(combined);
  }
  return ret;
}

Layer grad_layer(const Layer &layer) {
  // creat grad outputs tensors
  std::vector<te::Tensor> outputs;
  std::vector<te::Tensor> grad_outputs;
  for (const auto &op : layer->ops) {
    te::Tensor o = op.output(0);
    outputs.push_back(o);
    te::Tensor g = te::placeholder(o->shape, o->dtype, "grad_" + o->op->name);
    grad_outputs.push_back(g);
  }
  int num_outputs = (int)outputs.size();

  std::vector<Array<te::Tensor>> all_grad_weights;
  std::vector<Array<te::Tensor>> all_grad_inputs;
  for (int i = 0; i < num_outputs; ++i) {
    Array<te::Tensor> grad_weights =
        Gradient(outputs[i], layer->weights, grad_outputs[i]);
    Array<te::Tensor> grad_inputs =
        Gradient(outputs[i], layer->inputs, grad_outputs[i]);
    all_grad_weights.push_back(grad_weights);
    all_grad_inputs.push_back(grad_inputs);
  }

  Array<te::Tensor> grad_weights = combine_results(all_grad_weights);
  Array<te::Tensor> grad_inputs = combine_results(all_grad_inputs);

  // note that we put weights ahead of inputs
  // this is important to getting the grads and performs
  // gradient descent
  Array<te::Operation> ops;
  for (auto gw : grad_weights) {
    ops.push_back(gw->op);
  }
  for (auto gi : grad_inputs) {
    ops.push_back(gi->op);
  }

  Array<te::Tensor> inputs;
  for (auto inp : layer->inputs) {
    inputs.push_back(inp);
  }
  for (auto go : grad_outputs) {
    inputs.push_back(go);
  }

  return Layer("grad_" + layer->name, ops, inputs, layer->weights,
               layer->const_scalars, layer->const_tensors);
}

TVM_REGISTER_GLOBAL("ditto.autograd.GradLayer")
    .set_body_typed(grad_layer);

} // namespace autograd

} // namespace ditto