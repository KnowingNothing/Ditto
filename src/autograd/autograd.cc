#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <autograd/autodiff/autodiff.h>
#include <autograd/autograd.h>

#include <deque>
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

  // note that we put weights behind of inputs
  // this is important to getting the grads and performs
  // gradient descent
  Array<te::Operation> ops;
  for (auto gi : grad_inputs) {
    ops.push_back(gi->op);
  }
  for (auto gw : grad_weights) {
    ops.push_back(gw->op);
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

Graph grad_graph(const Graph &graph, bool reserve_forward) {
  Array<Layer> all_layers = graph->GetAllLayers();
  std::unordered_map<Layer, Layer> layer_map;
  std::unordered_map<Layer, std::vector<std::vector<std::pair<Layer, int>>>>
      feed_graph;
  int num_layers = (int)all_layers.size();
  CHECK(num_layers > 0);

  for (auto layer : all_layers) {
    Layer grad = grad_layer(layer);
    layer_map[layer] = grad;
    int num_inputs = (int)layer->input_layer_tensors_.size();
    for (int i = 0; i < num_inputs; ++i) {
      LayerTensor inp = layer->input_layer_tensors_[i];
      if (inp->layer.defined()) {
        if (!feed_graph.count(inp->layer)) {
          int num_outputs = (int)inp->layer->output_layer_tensors_.size();
          feed_graph[inp->layer].resize(num_outputs);
        }
        feed_graph[inp->layer][inp->value_idx].push_back(
            std::make_pair(layer, i));
      }
    }
  }

  std::unordered_set<Layer> visit;
  std::function<void(Layer layer)> helper;
  helper = [&](Layer layer) {
    if (visit.count(layer)) {
      return;
    }
    visit.insert(layer);
    CHECK(layer_map.count(layer));
    Layer grad = layer_map.at(layer);
    std::vector<LayerTensor> new_inputs;
    int num_old_inputs = (int)layer->input_layer_tensors_.size();
    for (int i = 0; i < num_old_inputs; ++i) {
      if (reserve_forward) {
        new_inputs.push_back(layer->input_layer_tensors_[i]);
      } else {
        LayerTensor org = layer->input_layer_tensors_[i];
        LayerTensor lt = LayerTensor(org->name, Layer(), org->tensor, 0);
        new_inputs.push_back(lt);
      }
    }

    if (!feed_graph.count(layer)) {
      // the last layer in original graph
      for (auto out : layer->ops) {
        te::Tensor output = out.output(0);
        te::Tensor t =
            te::placeholder(output->shape, output->dtype, "grad_" + out->name);
        LayerTensor lt = LayerTensor(t->op->name, Layer(), t, 0);
        new_inputs.push_back(lt);
      }
    } else {
      // layer has consumers in original graph
      auto vec1 = feed_graph.at(layer);
      // these two vectors record the same information
      std::vector<std::vector<LayerTensor>> grads_to_outputs;
      std::vector<Array<te::Tensor>> grads_tensors_to_outputs;
      // the grad to each output of layer
      Array<te::Tensor> combined_grads;

      for (auto vec2 : vec1) {
        // vec2: layers that consume one output
        // these two vectors stores the same information
        std::vector<LayerTensor> grad_to_this_output;
        Array<te::Tensor> grad_tensors_to_this_output;
        // this format so that we can use combine_result function
        // [        combine direction
        //  [grad1],    |
        //  [grad2],    |
        // ]            v
        std::vector<Array<te::Tensor>> grad_to_this_output_to_combine;
        for (auto kv : vec2) {
          helper(kv.first); // visit the consumer layer
          CHECK(layer_map.count(kv.first));
          Layer tmp_grad = layer_map.at(kv.first);
          // tmp_grad's output_layer_tensors_ should be updated
          CHECK(tmp_grad->output_layer_tensors_.size() > 0U);
          // output_layer_tensors_ = [List[grad_to_inputs],
          // List[grad_to_weights]] kv.second is the input ordinal number for
          // kv.first
          LayerTensor lt = tmp_grad->output_layer_tensors_[kv.second];
          te::Tensor t = te::placeholder(lt->tensor->shape, lt->tensor->dtype,
                                         lt->tensor->op->name + "_out");
          LayerTensor new_lt =
              LayerTensor(lt->name, lt->layer, t, lt->value_idx);
          grad_to_this_output.push_back(new_lt);
          grad_tensors_to_this_output.push_back(new_lt->tensor);
          grad_to_this_output_to_combine.push_back({new_lt->tensor});
        }
        grads_to_outputs.push_back(grad_to_this_output);
        grads_tensors_to_outputs.push_back(grad_tensors_to_this_output);
        // TODO: what if grad_to_this_output_to_combine.size() == 0?
        CHECK(grad_to_this_output_to_combine.size() > 0U)
            << "Layer " << layer->name
            << " has partial outputs consumed, while other outputs are not "
               "used.\n";
        Array<te::Tensor> combined =
            combine_results(grad_to_this_output_to_combine);
        CHECK(combined.size() == 1U);
        combined_grads.push_back(combined[0]);
      }

      // redundant check
      CHECK(grads_to_outputs.size() == grads_tensors_to_outputs.size());
      CHECK(grads_to_outputs.size() == combined_grads.size());
      // make up new layer for combined grads
      int num_combine_layers = (int)combined_grads.size();
      for (int i = 0; i < num_combine_layers; ++i) {
        if (grads_to_outputs[i].size() == 1U) {
          // no need to make new layer
          new_inputs.push_back(grads_to_outputs[i][0]);
        } else {
          // Array<te::Tensor> new_te_inputs;
          // std::vector<LayerTensor> new_tensors_inputs;
          // for (auto inp : grads_to_outputs[i]) {
          //   te::Tensor t = te::placeholder(inp->tensor->shape,
          //   inp->tensor->dtype, inp->tensor->op->name + "_out"); LayerTensor
          //   lt = LayerTensor(inp->name, inp->layer, t, inp->value_idx);
          //   new_te_inputs.push_back(t);
          //   new_tensors_inputs.push_back(lt);
          // }
          // need a new layer
          Layer combine_layer = Layer("combine", {combined_grads[i]->op},
                                      grads_tensors_to_outputs[i], {}, {}, {});

          // produce outputs
          std::vector<LayerTensor> new_outputs =
              combine_layer.ProduceOutputs(grads_to_outputs[i]);
          // redundant check
          CHECK(new_outputs.size() == 1U);
          new_inputs.push_back(new_outputs[0]);
        }
      }
    }
    // produce outputs
    std::vector<LayerTensor> new_outputs = grad.ProduceOutputs(new_inputs);
  };

  Array<LayerTensor> new_graph_inputs;
  Array<LayerTensor> new_graph_outputs;

  for (auto layer : all_layers) {
    helper(layer);
    Layer grad = layer_map.at(layer);
    int num_inputs = (int)layer->inputs.size();
    int num_weights = (int)layer->weights.size();
    int num_outputs = (int)layer->ops.size();

    // TODO: this assumes each layer is in either of two states:
    // 1. all outputs are used
    // 2. no output is used
    if (!feed_graph.count(layer)) {
      CHECK((int)grad->input_layer_tensors_.size() == num_inputs + num_outputs);
      for (int i = 0; i < num_outputs; ++i) {
        new_graph_inputs.push_back(grad->input_layer_tensors_[i + num_inputs]);
      }
    }

    CHECK((int)grad->output_layer_tensors_.size() == num_inputs + num_weights);
    for (int i = 0; i < num_weights; ++i) {
      new_graph_outputs.push_back(grad->output_layer_tensors_[i + num_inputs]);
    }
  }

  // the second pass to collect other graph inputs
  if (reserve_forward) {
    for (auto inp : graph->graph_inputs) {
      new_graph_inputs.push_back(inp);
    }
  } else {
    for (auto layer : all_layers) {
      Layer grad = layer_map.at(layer);
      int num_inputs = (int)layer->inputs.size();
      int num_outputs = (int)layer->ops.size();
      CHECK((int)grad->input_layer_tensors_.size() == num_inputs + num_outputs);
      for (int i = 0; i < num_inputs; ++i) {
        new_graph_inputs.push_back(grad->input_layer_tensors_[i]);
      }
    }
  }

  return Graph("grad_" + graph->name, new_graph_inputs, new_graph_outputs);
}

TVM_REGISTER_GLOBAL("ditto.autograd.GradLayer").set_body_typed(grad_layer);

TVM_REGISTER_GLOBAL("ditto.autograd.GradGraph").set_body_typed(grad_graph);

} // namespace autograd

} // namespace ditto