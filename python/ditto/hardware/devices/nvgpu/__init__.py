from .gpu_base import TensorCoreGPU
from .cuda_param import *

V100_16GB = TensorCoreGPU("V100-16GB", "sm70", V100_16GB_param)

V100_32GB = TensorCoreGPU("V100-32GB", "sm70", V100_32GB_param)

A100_40GB = TensorCoreGPU("A100-40GB", "sm80", A100_40GB_param)

A100_80GB = TensorCoreGPU("A100-80GB", "sm80", A100_80GB_param)

RTX3090 = TensorCoreGPU("RTX3090", "sm86", RTX3090_param)

SUPPORTED_GPUS = {
    "V100-16GB": V100_16GB,
    "V100-32GB": V100_32GB,
    "V100": V100_16GB,
    "A100-40GB": A100_40GB,
    "A100-80GB": A100_80GB,
    "A100": A100_40GB,
    "RTX3090": RTX3090
}


SUPPORTED_GPU_PARAMS = {
    "V100-16GB": V100_16GB_param,
    "V100-32GB": V100_32GB_param,
    "V100": V100_16GB_param,
    "A100-40GB": A100_40GB_param,
    "A100-80GB": A100_80GB_param,
    "A100": A100_40GB_param,
    "RTX3090": RTX3090_param
}


def query_gpu(device_name):
    """Use a string key to query the hardare device

    Args:
        target (str): the string key, e.g., V100-16GB

    Returns:
    ---
    ditto.hardware.HardwareDevice
    """
    return SUPPORTED_GPUS[device_name.upper()]


def query_gpu_param(device_name):
    """Use a string key to query the hardare device

    Args:
        target (str): the string key, e.g., V100-16GB

    Returns:
    ---
    ditto.hardware.HardwareParam
    """
    return SUPPORTED_GPU_PARAMS[device_name.upper()]
