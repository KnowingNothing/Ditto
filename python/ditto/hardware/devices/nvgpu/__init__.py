from .gpu_base import TensorCoreGPU
from .cuda_param import V100_16GB_param, V100_32GB_param

V100_16GB = TensorCoreGPU("V100-16GB", "sm70", V100_16GB_param)

V100_32GB = TensorCoreGPU("V100-32GB", "sm70", V100_32GB_param)

SUPPORTED_GPUS = {
    "V100-16GB": V100_16GB,
    "V100-32GB": V100_32GB,
    "V100": V100_16GB,
}


SUPPORTED_GPU_PARAMS = {
    "V100-16GB": V100_16GB_param,
    "V100-32GB": V100_32GB_param,
    "V100": V100_16GB_param,
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
