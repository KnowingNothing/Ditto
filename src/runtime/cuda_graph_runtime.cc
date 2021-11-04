#include <tvm/runtime/data_type.h>

#include <runtime/cuda_graph_runtime.h>

#include <chrono>
#include <sstream>
#include <unordered_set>

namespace ditto {

namespace runtime {

void CUDAGraphEngine::Capture() {
  this->Run();
  CHECKCUDA(cudaStreamBeginCapture(static_cast<cudaStream_t>(capture_stream_),
                                   cudaStreamCaptureModeGlobal));
  this->Run();

  cudaGraph_t graph;
  CHECKCUDA(
      cudaStreamEndCapture(static_cast<cudaStream_t>(capture_stream_), &graph));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  CHECKCUDA(cudaGraphGetNodes(graph, nodes, &numNodes));
  LOG(INFO) << "Num of nodes in cuda garph = " << numNodes << ".\n";
  CHECKCUDA(cudaGraphInstantiate(&cuda_graph_exec_, graph, NULL, NULL, 0));
  captured_ = true;
}

void CUDAGraphEngine::GraphRun() {
  if (!captured_)
    this->Capture();
  cudaStream_t stream = static_cast<cudaStream_t>(capture_stream_);
  CHECKCUDA(cudaGraphLaunch(cuda_graph_exec_, stream));
  CHECKCUDA(cudaStreamSynchronize(stream));
}

FloatImm CUDAGraphEngine::TimeIt(int number) {
  // warm up
  if (!captured_)
    this->Capture();
  cudaStream_t stream = static_cast<cudaStream_t>(capture_stream_);
  CHECKCUDA(cudaGraphLaunch(cuda_graph_exec_, stream));
  CHECKCUDA(cudaStreamSynchronize(stream));
  auto beg = std::chrono::steady_clock::now();
  for (int i = 0; i < number; ++i) {
    CHECKCUDA(cudaGraphLaunch(cuda_graph_exec_, stream));
  }
  CHECKCUDA(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  double execution_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() /
      number / 1e3;
  return FloatImm(tvm::runtime::DataType::Float(32), float(execution_time));
}

PackedFunc CUDAGraphEngine::GetFunction(const std::string &name,
                                        const ObjectPtr<Object> &sptr_to_self) {
  if (name == "capture") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->Capture();
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      this->GraphRun();
    });
  } else if (name == "timeit") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue *rv) {
      *rv = this->TimeIt(args[0]);
    });
  } else {
    return GraphEngine::GetFunction(name, sptr_to_self);
  }
}

TVM_REGISTER_GLOBAL("ditto.runtime.create_cuda_graph_engine")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
      auto exec = make_object<CUDAGraphEngine>(args[0], args[1], args[2]);
      *rv = Module(exec);
    });

} // namespace runtime

} // namespace ditto