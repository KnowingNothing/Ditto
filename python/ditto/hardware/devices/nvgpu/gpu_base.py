from ...hw_compute import *
from ...hw_memory import *
from ...hw_path import *
from ...hw_param import HardwareParam
from .cuda_isa import *
from .cuda_pattern import *
from .cuda_path import *


def convert_sm_to_int(sm_arch: str):
    assert sm_arch[:2] == "sm"
    return int(sm_arch[2:])


def TensorCoreGPU(name: str, sm_arch: int, param: HardwareParam):
    if isinstance(sm_arch, str):
        sm_arch = convert_sm_to_int(sm_arch)
    assert (
        sm_arch >= 70
    ), f"Tensor core GPU is at least is sm70, but given sm{sm_arch}.\n"
    if sm_arch < 75:
        tc_isa = wmma_sm70_isa
        tc_pattern = wmma_sm70_patterns
        tc_path = wmma_sm70_paths
    elif sm_arch < 80:
        tc_isa = wmma_sm75_isa
        tc_pattern = wmma_sm75_patterns
        tc_path = wmma_sm75_paths
    elif sm_arch < 90:
        tc_isa = wmma_sm80_isa
        tc_pattern = wmma_sm80_patterns
        tc_path = wmma_sm80_paths
    else:
        raise ValueError(f"No support for sm{sm_arch} tensor core GPU.\n")
    cuda_core_isa = {x.name: x for x in all_scalar_isa}
    tensor_core_isa = {x.name: x for x in tc_isa}
    cuda_core = hw_unit(cuda_core_isa, name="cuda_core")
    tensor_core = hw_unit(tensor_core_isa, name="tensor_core")

    register_patterns = {**scalar_patterns, **tc_pattern}
    register = hw_local_mem(32, register_patterns, name="register")

    topology = {cuda_core: {register: scalar_paths}, tensor_core: {register: tc_path}}

    processor = hw_heteroprocessor(
        [cuda_core, tensor_core], [register], topology, name="cuda_processor"
    )

    shared_mem = hw_shared_mem(
        param.shared_memory_per_group_kb, scalar_patterns, name="shared_memory"
    )

    group = hw_homogroup(
        processor,
        shared_mem,
        param.num_processors_per_group,
        name="streaming_multiprocessors",
    )

    global_mem = hw_global_mem(
        param.global_memory_gb, scalar_patterns, name="global_memory"
    )

    device = hw_device(group, global_mem, param.num_groups, name=name)
    return device
