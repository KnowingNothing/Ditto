#pragma once

#include <cuda_runtime.h>
#include <runtime/graph_runtime.h>

namespace ditto {
using namespace tvm;
using namespace tvm::runtime;
using namespace ditto::auto_compute;
namespace runtime {

#define CHECKCUDA(func)                                                        \
  {                                                                            \
    cudaError_t status = (func);                                               \
    CHECK(status == cudaSuccess || status == cudaErrorCudartUnloading)         \
        << "CUDA: " << cudaGetErrorString(status);                             \
  }

class CUDAGraphEngine : public GraphEngine {
public:
  CUDAGraphEngine(Graph graph, Map<String, Module> built_mods, Device dev)
      : GraphEngine(graph, built_mods, dev), captured_(false) {
    TVMStreamCreate(dev_.device_type, dev_.device_id, &capture_stream_);
    TVMSetStream(dev_.device_type, dev_.device_id, capture_stream_);
  }

  void Capture();

  void GraphRun();

  FloatImm TimeIt(int number);

  PackedFunc GetFunction(const std::string &name,
                         const ObjectPtr<Object> &sptr_to_self);

private:
  TVMStreamHandle capture_stream_{0};
  cudaGraphExec_t cuda_graph_exec_;
  bool captured_{false};
};

} // namespace runtime

} // namespace ditto