#pragma once

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target.h>

#include <auto_compute/graph.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace ditto {
using namespace tvm;
using namespace tvm::runtime;
using namespace ditto::auto_compute;
namespace runtime {

class TVM_DLL GraphEngine : public ModuleNode {
public:
  virtual PackedFunc GetFunction(const std::string &name,
                                 const ObjectPtr<Object> &sptr_to_self);

  const char *type_key() const final { return "ditto.runtime.GraphEngine"; }

  void Init(Graph graph, Map<String, Module> built_mods, Device dev);

  void SetInputs(Array<NDArray> inputs);

  void SetWeight(Layer layer, te::Tensor t, NDArray data);

  void Compile();

  void Run();

  FloatImm TimeIt(int number = 100);

  Array<NDArray> GetOutputs();

  class Arguments {
    public:
     std::vector<TVMValue> arg_values;
     std::vector<int> arg_tcodes;
     std::vector<NDArray> args;
     std::vector<int64_t> shape_data;
  };

private:
  Graph graph_;
  Device dev_;
  Map<String, Module> built_mods_;
  std::vector<std::string> exe_seq_;
  std::unordered_map<std::string, Layer> workloads_;
  std::unordered_map<Layer, std::vector<NDArray>> input_buffer_;
  std::unordered_map<Layer, std::vector<NDArray>> weight_buffer_;
  std::unordered_map<Layer, std::vector<NDArray>> output_buffer_;
  std::unordered_map<std::string, PackedFunc> built_funcs_;
  std::vector<std::function<void()>> to_exe_;

  std::unordered_map<te::Tensor, std::vector<std::pair<Layer, int>>>
      feed_layers_;
};

} // namespace runtime

} // namespace ditto