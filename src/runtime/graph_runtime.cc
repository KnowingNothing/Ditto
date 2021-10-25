#include <tvm/runtime/data_type.h>

#include <runtime/graph_runtime.h>
#include <utils/unpack_func.h>

#include <chrono>
#include <sstream>
#include <unordered_set>

namespace ditto {

namespace runtime {

PackedFunc GraphEngine::GetFunction(const std::string &name,
                                    const ObjectPtr<Object> &sptr_to_self) {
  if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->Init(args[0], args[1], args[2]);
    });
  } else if (name == "set_inputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->SetInputs(args[0]);
    });
  } else if (name == "set_weight") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->SetWeight(args[0], args[1], args[2]);
    });
  } else if (name == "compile") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->Compile();
    });
  } else if (name == "run") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue *rv) { this->Run(); });
  } else if (name == "timeit") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      *rv = this->TimeIt(args[0]);
    });
  } else if (name == "get_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      *rv = this->GetOutputs();
    });
  } else {
    return PackedFunc();
  }
}

FloatImm make_float(float value) {
  return FloatImm(tvm::runtime::DataType::Float(32), value);
}

void GraphEngine::Init(Graph graph, Map<String, Module> built_mods,
                       Device dev) {
  graph_ = graph;
  built_mods_ = built_mods;
  dev_ = dev;
  exe_seq_.clear();
  workloads_.clear();
  input_buffer_.clear();
  weight_buffer_.clear();
  output_buffer_.clear();
  feed_layers_.clear();

  const auto *get_random =
      tvm::runtime::Registry::Get("runtime.ndarray.random");
  CHECK(get_random != nullptr) << "runtime.ndarray.random.\n";

  for (auto layer : graph->GetAllLayers()) {
    std::string wkl_key = layer->GetFingerprint();
    exe_seq_.push_back(wkl_key);
    workloads_[wkl_key] = layer;
    int count_inp = 0;
    for (auto inp : layer->input_layer_tensors_) {
      feed_layers_[inp->tensor].push_back(std::make_pair(layer, count_inp));
      if (inp->layer.defined()) {
        std::string inp_fin = inp->layer->GetFingerprint();
        CHECK(output_buffer_.count(inp_fin))
            << "Can't find layer " << inp->layer << " in input_buffer_.\n";
        CHECK((int)output_buffer_.at(inp_fin).size() > inp->value_idx);
        input_buffer_[wkl_key].push_back(
            output_buffer_[inp_fin][inp->value_idx]);
      } else {
        std::ostringstream oss;
        oss << inp->tensor->dtype;
        NDArray ary = (*get_random)(inp->tensor->shape, make_float(-10.0f),
                                    make_float(10.0f), oss.str(), dev_);
        input_buffer_[wkl_key].push_back(ary);
      }
      count_inp += 1;
    }
    for (auto w : layer->weights) {
      std::ostringstream oss;
      oss << w->dtype;
      NDArray ary = (*get_random)(w->shape, make_float(-10.0f),
                                  make_float(10.0f), oss.str(), dev_);
      weight_buffer_[wkl_key].push_back(ary);
    }
    for (auto op : layer->ops) {
      te::Tensor output = op.output(0);
      std::ostringstream oss;
      oss << output->dtype;
      NDArray ary = (*get_random)(output->shape, make_float(-10.0f),
                                  make_float(10.0f), oss.str(), dev_);
      output_buffer_[wkl_key].push_back(ary);
    }
  }
}

void GraphEngine::SetInputs(Array<NDArray> inputs) {
  CHECK(inputs.size() == graph_->graph_inputs.size());
  int count_inp = 0;
  for (auto inp : graph_->graph_inputs) {
    CHECK(feed_layers_.count(inp->tensor));
    for (auto kv : feed_layers_.at(inp->tensor)) {
      Layer layer = kv.first;
      std::string wkl_key = layer->GetFingerprint();
      int id = kv.second;
      CHECK(input_buffer_.count(wkl_key));
      CHECK(id < (int)input_buffer_[wkl_key].size());
      input_buffer_[wkl_key][id] = inputs[count_inp];
    }

    count_inp += 1;
  }
}

void GraphEngine::SetWeight(Layer layer, te::Tensor t, NDArray data) {
  int id = -1;
  for (auto w : layer->weights) {
    id += 1;
    if (t == w) {
      break;
    }
  }
  CHECK(id >= 0 && id < (int)layer->weights.size());
  std::string wkl_key = layer->GetFingerprint();
  CHECK(weight_buffer_.count(wkl_key));
  weight_buffer_[wkl_key][id] = data;
}

void GraphEngine::Compile() {
  built_funcs_.clear();
  to_exe_.clear();
  for (auto kv : built_mods_) {
    PackedFunc func = kv.second->GetFunction("default_function");
    CHECK(func != nullptr) << "Can't find default_function in module.\n";
    std::shared_ptr<GraphEngine::Arguments> arg_ptr =
        std::make_shared<GraphEngine::Arguments>();
    // std::vector<TVMValue> args;
    // std::vector<NDArray> arg_values;
    // std::vector<int> arg_tcodes;
    std::string wkl_key = std::string(kv.first);
    CHECK(workloads_.count(wkl_key))
        << "Can't find workload " << wkl_key << ".\n";
    if (input_buffer_.count(wkl_key)) {
      for (auto inp : input_buffer_.at(wkl_key)) {
        TVMValue v;
        // DLTensor t(*inp.operator->());
        v.v_handle = &inp;
        arg_ptr->args.push_back(inp);
        CHECK(static_cast<NDArray *>(v.v_handle) != nullptr);
        arg_ptr->arg_values.push_back(v);
        arg_ptr->arg_tcodes.push_back(kTVMNDArrayHandle);
      }
    }
    if (weight_buffer_.count(wkl_key)) {
      for (auto w : weight_buffer_.at(wkl_key)) {
        TVMValue v;
        // DLTensor t(*w.operator->());
        v.v_handle = &w;
        arg_ptr->args.push_back(w);
        arg_ptr->arg_values.push_back(v);
        arg_ptr->arg_tcodes.push_back(kTVMNDArrayHandle);
      }
    }
    if (output_buffer_.count(wkl_key)) {
      for (auto out : output_buffer_.at(wkl_key)) {
        TVMValue v;
        // DLTensor t(*out.operator->());
        v.v_handle = &out;
        arg_ptr->args.push_back(out);
        arg_ptr->arg_values.push_back(v);
        arg_ptr->arg_tcodes.push_back(kTVMNDArrayHandle);
      }
    }
    auto fexe = [func, arg_ptr, wkl_key]() {
      //   TVMRetValue rv;
      //   TVMArgs targs(arg_ptr->arg_values.data(), arg_ptr->arg_tcodes.data(),
      //   (int)(arg_ptr->arg_values.size()));
      auto *call_unpack = new utils::CallFunc<tvm::runtime::PackedFunc,
                                              tvm::runtime::NDArray>();
      // func.CallPacked(targs, &rv);
      (*call_unpack)(func, arg_ptr->args);
    };
    built_funcs_[wkl_key] = fexe;
  }

  for (auto key : exe_seq_) {
    CHECK(built_funcs_.count(key));
    to_exe_.push_back(built_funcs_.at(key));
  }
}

void GraphEngine::Run() {
  for (auto f : to_exe_) {
    f();
  }
}

FloatImm GraphEngine::TimeIt(int number) {
  // warm up
  this->Run();
  tvm::runtime::DeviceAPI::Get(dev_)->StreamSync(dev_, nullptr);
  auto beg = std::chrono::steady_clock::now();
  for (int i = 0; i < number; ++i) {
    this->Run();
  }
  tvm::runtime::DeviceAPI::Get(dev_)->StreamSync(dev_, nullptr);
  auto end = std::chrono::steady_clock::now();
  double execution_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() /
      number / 1e3;
  return make_float(float(execution_time));
}

Array<NDArray> GraphEngine::GetOutputs() {
  Array<NDArray> ret;
  for (auto out : graph_->graph_outputs) {
    if (out->layer.defined()) {
      Layer layer = out->layer;
      std::string wkl_key = layer->GetFingerprint();
      int id = out->value_idx;
      CHECK(output_buffer_.count(wkl_key));
      CHECK(id < (int)output_buffer_[wkl_key].size());
      ret.push_back(output_buffer_[wkl_key][id]);
    }
  }
  return ret;
}

TVM_REGISTER_GLOBAL("ditto.runtime.create_graph_engine")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
      auto exec = make_object<GraphEngine>();
      exec->Init(args[0], args[1], args[2]);
      *rv = Module(exec);
    });

} // namespace runtime

} // namespace ditto