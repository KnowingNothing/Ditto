from .config import V100
from .param import HardwareParam


SUPPORTED_TARGETS = {
    "gpu": {
        "cuda": {
            "V100": V100
        }
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


def query_hw_param(target):
    """Use a string key to query the hardare params

    Args:
        target (str): the string key, e.g., cuda

    Returns:
    ---
    ditto.hardware.HardwareParam
    """
    dev_type, pmodel, arch = parse_target(target)
    return SUPPORTED_TARGETS[dev_type][pmodel][arch]
