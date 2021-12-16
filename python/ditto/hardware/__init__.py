from .config import V100


def query_hw_param(target):
    """Use a string key to query the hardare params

    Args:
        target (str): the string key, e.g., cuda

    Returns:
    ---
    ditto.hardware.HardwareParam
    """
    raise NotImplementedError()
