import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object
from .hw_base import Hardware


@tvm._ffi.register_object("ditto.hardware.HardwareMemory")
class HardwareMemory(Hardware):
    """HardwareMemory object"""

    def __str__(self) -> str:
        ret = f"HardwareMemory({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.LocalMemory")
class LocalMemory(HardwareMemory):
    """Hardware local memory object"""

    def __str__(self) -> str:
        ret = f"LocalMemory({self.name}, {self.kb}KB)"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_local_mem(kb_size, patterns, name="hw_local_mem"):
    """The hardware local memory

    Args:
        kb_size (float): capacity in kb
        patterns (Map[String, hardware.Pattern]): supported access patterns
        name (str, optional): the name of the memory. Defaults to "hw_local_mem".

    Returns:
        hardware.LocalMemory: the hardware local memory
    """
    return _ffi_api.LocalMemory(name, kb_size, patterns)


@tvm._ffi.register_object("ditto.hardware.SharedMemory")
class SharedMemory(HardwareMemory):
    """Hardware shared memory object"""

    def __str__(self) -> str:
        ret = f"SharedMemory({self.name}, {self.kb}KB)"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_shared_mem(kb_size, patterns, name="hw_shared_mem"):
    """The hardware shared memory

    Args:
        kb_size (float): capacity in kb
        patterns (Map[String, hardware.Pattern]): supported access patterns
        name (str, optional): the name of the memory. Defaults to "hw_local_mem".

    Returns:
        hardware.SharedMemory: the hardware shared memory
    """
    return _ffi_api.SharedMemory(name, kb_size, patterns)


@tvm._ffi.register_object("ditto.hardware.GlobalMemory")
class GlobalMemory(HardwareMemory):
    """Hardware global memory object"""

    def __str__(self) -> str:
        ret = f"GlobalMemory({self.name}, {self.kb/1e6}GB)"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_global_mem(gb_size, patterns, name="hw_global_mem"):
    """The hardware global memory

    Args:
        gb_size (float): capacity in GB
        patterns (Map[String, hardware.Pattern]): supported access patterns
        name (str, optional): the name of the memory. Defaults to "hw_local_mem".

    Returns:
        hardware.GlobalMemory: the hardware global memory
    """
    return _ffi_api.GlobalMemory(name, gb_size, patterns)
