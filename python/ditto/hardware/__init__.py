from .hw_compute import *
from .hw_memory import *
from .hw_path import *
from . import pattern
from . import visa
from .devices.nvgpu import (
    query_gpu as query_nvgpu,
    query_gpu_param as query_nvgpu_param
)
from .hw_param import HardwareParam


QUERY_FUNCS = {
    "gpu": {
        "cuda": query_nvgpu
    }
}

QUERY_PARAM_FUNCS = {
    "gpu": {
        "cuda": query_nvgpu_param
    }
}


def parse_target(target):
    """Split the target string.
        Expect target format:
        target ::= device_type.programming_model.architecture
        device_type ::= gpu | cpu | npu | fpga | asic
        programming_model ::= cuda | opencl | c | llvm | cce | hls_c | hls_cl
        architecture ::= P100 | V100 | A100 | Xeon-Silver-4210R | Ascend-910B
    """
    return target.split(".")


def query_hw(target):
    """Use a string key to query the hardare params

    Args:
        target (str): the string key, e.g., gpu.cuda.V100-16GB

    Returns:
    ---
    ditto.hardware.HardwareParam
    """
    dev_type, pmodel, arch = parse_target(target)
    return QUERY_FUNCS[dev_type][pmodel](arch)


def query_hw_param(target):
    """Use a string key to query the hardare params

    Args:
        target (str): the string key, e.g., gpu.cuda.V100-16GB

    Returns:
    ---
    ditto.hardware.HardwareParam
    """
    dev_type, pmodel, arch = parse_target(target)
    return QUERY_PARAM_FUNCS[dev_type][pmodel](arch)
