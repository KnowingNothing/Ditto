#include <hardware/compute/device/nvgpu.h>

namespace ditto {

namespace hardware {

HardwareDevice NvSimtGPU(String sm_arch) {
  if (sm_arch == "sm61") {
    Map<String, ISA> isa_list;
    {
      /* binary add */
      // binary add fp32 fp32 fp32
      isa_list.Set(
          "cuda.add.fp32.fp32.fp32",
          ScalarBinaryAdd("cuda.add.fp32.fp32.fp32",
                          NvGPULatencyTable.at("cuda.add.fp32.fp32.fp32"),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32)));
      // binary add fp16 fp16 fp32
      isa_list.Set(
          "cuda.add.fp16.fp16.fp32",
          ScalarBinaryAdd("cuda.add.fp16.fp16.fp32",
                          NvGPULatencyTable.at("cuda.add.fp16.fp16.fp32"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(32)));
      // binary add fp16 fp16 fp16
      isa_list.Set(
          "cuda.add.fp16.fp16.fp16",
          ScalarBinaryAdd("cuda.add.fp16.fp16.fp16",
                          NvGPULatencyTable.at("cuda.add.fp16.fp16.fp16"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16)));
      // binary add int32 int32 int32
      isa_list.Set("cuda.add.int32.int32.int32",
                   ScalarBinaryAdd(
                       "cuda.add.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.add.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // binary add int8 int8 int32
      isa_list.Set(
          "cuda.add.int8.int8.int32",
          ScalarBinaryAdd("cuda.add.int8.int8.int32",
                          NvGPULatencyTable.at("cuda.add.int8.int8.int32"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(32)));
      // binary add int8 int8 int8
      isa_list.Set(
          "cuda.add.int8.int8.int8",
          ScalarBinaryAdd("cuda.add.int8.int8.int8",
                          NvGPULatencyTable.at("cuda.add.int8.int8.int8"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(8)));

      /* binary sub */
      // binary sub fp32 fp32 fp32
      isa_list.Set(
          "cuda.sub.fp32.fp32.fp32",
          ScalarBinarySub("cuda.sub.fp32.fp32.fp32",
                          NvGPULatencyTable.at("cuda.sub.fp32.fp32.fp32"),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32)));
      // binary sub fp16 fp16 fp32
      isa_list.Set(
          "cuda.sub.fp16.fp16.fp32",
          ScalarBinarySub("cuda.sub.fp16.fp16.fp32",
                          NvGPULatencyTable.at("cuda.sub.fp16.fp16.fp32"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(32)));
      // binary sub fp16 fp16 fp16
      isa_list.Set(
          "cuda.sub.fp16.fp16.fp16",
          ScalarBinarySub("cuda.sub.fp16.fp16.fp16",
                          NvGPULatencyTable.at("cuda.sub.fp16.fp16.fp16"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16)));
      // binary sub int32 int32 int32
      isa_list.Set("cuda.sub.int32.int32.int32",
                   ScalarBinarySub(
                       "cuda.sub.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.sub.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // binary sub int8 int8 int32
      isa_list.Set(
          "cuda.sub.int8.int8.int32",
          ScalarBinarySub("cuda.sub.int8.int8.int32",
                          NvGPULatencyTable.at("cuda.sub.int8.int8.int32"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(32)));
      // binary sub int8 int8 int8
      isa_list.Set(
          "cuda.sub.int8.int8.int8",
          ScalarBinarySub("cuda.sub.int8.int8.int8",
                          NvGPULatencyTable.at("cuda.sub.int8.int8.int8"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(8)));

      /* binary mul */
      // binary mul fp32 fp32 fp32
      isa_list.Set(
          "cuda.mul.fp32.fp32.fp32",
          ScalarBinaryMul("cuda.mul.fp32.fp32.fp32",
                          NvGPULatencyTable.at("cuda.mul.fp32.fp32.fp32"),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32)));
      // binary mul fp16 fp16 fp32
      isa_list.Set(
          "cuda.mul.fp16.fp16.fp32",
          ScalarBinaryMul("cuda.mul.fp16.fp16.fp32",
                          NvGPULatencyTable.at("cuda.mul.fp16.fp16.fp32"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(32)));
      // binary mul fp16 fp16 fp16
      isa_list.Set(
          "cuda.mul.fp16.fp16.fp16",
          ScalarBinaryMul("cuda.mul.fp16.fp16.fp16",
                          NvGPULatencyTable.at("cuda.mul.fp16.fp16.fp16"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16)));
      // binary mul int32 int32 int32
      isa_list.Set("cuda.mul.int32.int32.int32",
                   ScalarBinaryMul(
                       "cuda.mul.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.mul.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // binary mul int8 int8 int32
      isa_list.Set(
          "cuda.mul.int8.int8.int32",
          ScalarBinaryMul("cuda.mul.int8.int8.int32",
                          NvGPULatencyTable.at("cuda.mul.int8.int8.int32"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(32)));
      // binary mul int8 int8 int8
      isa_list.Set(
          "cuda.mul.int8.int8.int8",
          ScalarBinaryMul("cuda.mul.int8.int8.int8",
                          NvGPULatencyTable.at("cuda.mul.int8.int8.int8"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(8)));

      /* binary div */
      // binary div fp32 fp32 fp32
      isa_list.Set(
          "cuda.div.fp32.fp32.fp32",
          ScalarBinaryDiv("cuda.div.fp32.fp32.fp32",
                          NvGPULatencyTable.at("cuda.div.fp32.fp32.fp32"),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32)));
      // binary div fp16 fp16 fp32
      isa_list.Set(
          "cuda.div.fp16.fp16.fp32",
          ScalarBinaryDiv("cuda.div.fp16.fp16.fp32",
                          NvGPULatencyTable.at("cuda.div.fp16.fp16.fp32"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(32)));
      // binary div fp16 fp16 fp16
      isa_list.Set(
          "cuda.div.fp16.fp16.fp16",
          ScalarBinaryDiv("cuda.div.fp16.fp16.fp16",
                          NvGPULatencyTable.at("cuda.div.fp16.fp16.fp16"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16)));
      // binary div int32 int32 int32
      isa_list.Set("cuda.div.int32.int32.int32",
                   ScalarBinaryDiv(
                       "cuda.div.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.div.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // binary div int8 int8 int32
      isa_list.Set(
          "cuda.div.int8.int8.int32",
          ScalarBinaryDiv("cuda.div.int8.int8.int32",
                          NvGPULatencyTable.at("cuda.div.int8.int8.int32"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(32)));
      // binary div int8 int8 int8
      isa_list.Set(
          "cuda.div.int8.int8.int8",
          ScalarBinaryDiv("cuda.div.int8.int8.int8",
                          NvGPULatencyTable.at("cuda.div.int8.int8.int8"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(8)));

      /* binary mod */
      // binary mod fp32 fp32 fp32
      isa_list.Set(
          "cuda.mod.fp32.fp32.fp32",
          ScalarBinaryMod("cuda.mod.fp32.fp32.fp32",
                          NvGPULatencyTable.at("cuda.mod.fp32.fp32.fp32"),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32),
                          runtime::DataType::Float(32)));
      // binary mod fp16 fp16 fp32
      isa_list.Set(
          "cuda.mod.fp16.fp16.fp32",
          ScalarBinaryMod("cuda.mod.fp16.fp16.fp32",
                          NvGPULatencyTable.at("cuda.mod.fp16.fp16.fp32"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(32)));
      // binary mod fp16 fp16 fp16
      isa_list.Set(
          "cuda.mod.fp16.fp16.fp16",
          ScalarBinaryMod("cuda.mod.fp16.fp16.fp16",
                          NvGPULatencyTable.at("cuda.mod.fp16.fp16.fp16"),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16),
                          runtime::DataType::Float(16)));
      // binary mod int32 int32 int32
      isa_list.Set("cuda.mod.int32.int32.int32",
                   ScalarBinaryMod(
                       "cuda.mod.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.mod.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // binary mod int8 int8 int32
      isa_list.Set(
          "cuda.mod.int8.int8.int32",
          ScalarBinaryMod("cuda.mod.int8.int8.int32",
                          NvGPULatencyTable.at("cuda.mod.int8.int8.int32"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(32)));
      // binary mod int8 int8 int8
      isa_list.Set(
          "cuda.mod.int8.int8.int8",
          ScalarBinaryMod("cuda.mod.int8.int8.int8",
                          NvGPULatencyTable.at("cuda.mod.int8.int8.int8"),
                          runtime::DataType::Int(8), runtime::DataType::Int(8),
                          runtime::DataType::Int(8)));

      /* multiply-add */
      // multiply-add fp32 fp32 fp32
      isa_list.Set(
          "cuda.fma.fp32.fp32.fp32",
          ScalarMultiplyAdd("cuda.fma.fp32.fp32.fp32",
                            NvGPULatencyTable.at("cuda.fma.fp32.fp32.fp32"),
                            runtime::DataType::Float(32),
                            runtime::DataType::Float(32),
                            runtime::DataType::Float(32)));
      // multiply-add fp16 fp16 fp32
      isa_list.Set(
          "cuda.fma.fp16.fp16.fp32",
          ScalarMultiplyAdd("cuda.fma.fp16.fp16.fp32",
                            NvGPULatencyTable.at("cuda.fma.fp16.fp16.fp32"),
                            runtime::DataType::Float(16),
                            runtime::DataType::Float(16),
                            runtime::DataType::Float(32)));
      // multiply-add fp16 fp16 fp16
      isa_list.Set(
          "cuda.fma.fp16.fp16.fp16",
          ScalarMultiplyAdd("cuda.fma.fp16.fp16.fp16",
                            NvGPULatencyTable.at("cuda.fma.fp16.fp16.fp16"),
                            runtime::DataType::Float(16),
                            runtime::DataType::Float(16),
                            runtime::DataType::Float(16)));
      // multiply-add int32 int32 int32
      isa_list.Set("cuda.fma.int32.int32.int32",
                   ScalarMultiplyAdd(
                       "cuda.fma.int32.int32.int32",
                       NvGPULatencyTable.at("cuda.fma.int32.int32.int32"),
                       runtime::DataType::Int(32), runtime::DataType::Int(32),
                       runtime::DataType::Int(32)));
      // multiply-add int8 int8 int32
      isa_list.Set("cuda.fma.int8.int8.int32",
                   ScalarMultiplyAdd(
                       "cuda.fma.int8.int8.int32",
                       NvGPULatencyTable.at("cuda.fma.int8.int8.int32"),
                       runtime::DataType::Int(8), runtime::DataType::Int(8),
                       runtime::DataType::Int(32)));
      // multiply-add int8 int8 int8
      isa_list.Set("cuda.fma.int8.int8.int8",
                   ScalarMultiplyAdd(
                       "cuda.fma.int8.int8.int8",
                       NvGPULatencyTable.at("cuda.fma.int8.int8.int8"),
                       runtime::DataType::Int(8), runtime::DataType::Int(8),
                       runtime::DataType::Int(8)));
    }

    HardwareUnit cuda_core("CudaCore", isa_list);
    NvGPUParam param;
    Map<String, Pattern> pattern_list;
    int num_reg = param->GetNum32BitRegistersPerSM(sm_arch);
    int num_proc_per_SM = param->GetNumProcessorPerSM(sm_arch);
    // 32bit = 4 byte, 1 kb = 1024 byte
    double reg_kb = num_reg / num_proc_per_SM * 4 / 1024.0;
    {
      pattern_list.Set(
          "scalar_int32",
          ScalarPattern("scalar_int32", runtime::DataType::Int(32), "local"));
      pattern_list.Set(
          "scalar_int8",
          ScalarPattern("scalar_int8", runtime::DataType::Int(8), "local"));
      pattern_list.Set(
          "scalar_uint8",
          ScalarPattern("scalar_uint8", runtime::DataType::UInt(8), "local"));
      pattern_list.Set(
          "scalar_fp32",
          ScalarPattern("scalar_fp32", runtime::DataType::Float(32), "local"));
      pattern_list.Set(
          "scalar_fp16",
          ScalarPattern("scalar_fp16", runtime::DataType::Float(16), "local"));
      pattern_list.Set(
          "scalar_fp64",
          ScalarPattern("scalar_fp64", runtime::DataType::Float(64), "local"));
    }
    LocalMemory reg("CudaRegister", reg_kb, pattern_list);
    Topology topology;
    Array<HardwarePath> paths;
    {
      paths.push_back(ComputePath(isa_list.at("cuda.add.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.add.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));

      paths.push_back(ComputePath(isa_list.at("cuda.sub.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.sub.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));

      paths.push_back(ComputePath(isa_list.at("cuda.mul.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mul.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));

      paths.push_back(ComputePath(isa_list.at("cuda.div.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.div.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));

      paths.push_back(ComputePath(isa_list.at("cuda.mod.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), None(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.mod.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));

      paths.push_back(ComputePath(isa_list.at("cuda.fma.fp32.fp32.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.fp16.fp16.fp32"),
                                  pattern_list.at("scalar_fp32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.fp16.fp16.fp16"),
                                  pattern_list.at("scalar_fp16"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.int32.int32.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.int8.int8.int32"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  None()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.int8.int8.int32"),
                                  pattern_list.at("scalar_int32"), Direct(),
                                  Direct()));
      paths.push_back(ComputePath(isa_list.at("cuda.fma.int8.int8.int8"),
                                  pattern_list.at("scalar_int8"), Direct(),
                                  Direct()));
    }
    Map<LocalMemory, Array<HardwarePath>> mem_paths;
    mem_paths.Set(reg, paths);
    topology.Set(cuda_core, mem_paths);
    HeteroProcessor processor("CudaProcessor", {cuda_core}, {reg}, topology);
    SharedMemory smem("CudaSmem", param->GetKBSharedMemPerSM(sm_arch),
                      pattern_list);
    HomoGroup sm("CudaSM", processor, smem, num_proc_per_SM, 1, 1);
    GlobalMemory gm("CudaGlobalMem", param->GetGBGlobalMemPerSM(sm_arch),
                    pattern_list);
    HardwareDevice P100("P100", sm, gm, param->GetNumSMPerGPU(sm_arch), 1, 1);
    return P100;
  } else {
    CHECK(false) << "No support for Nvidia Simt " << sm_arch << " GPU.\n";
    return HardwareDevice();
  }
}

} // namespace hardware

} // namespace ditto