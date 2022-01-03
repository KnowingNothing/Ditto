import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.HardwarePath")
class HardwarePath(Object):
    """HardwarePath object"""


@tvm._ffi.register_object("ditto.hardware.ComputePath")
class ComputePath(HardwarePath):
    """Hardware compute path object"""

    def __str__(self) -> str:
        ret = f"ComputePath({self.isa}, {self.pattern}, {self.load}, {self.store})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def compute_path(isa, pattern, load, store):
    """The compute path

    Args:
        isa (hardware.ISA): the compute isa
        pattern (hardware.Pattern): the memory pattern
        load (hardware.ISA): the load isa
        store (hardware.ISA): the store isa

    Returns:
        ComputePath: the compute path
    """
    return _ffi_api.ComputePath(isa, pattern, load, store)


@tvm._ffi.register_object("ditto.hardware.DataPath")
class DataPath(HardwarePath):
    """Hardware data path object"""

    def __str__(self) -> str:
        ret = f"DataPath({self.isa}, {self.src_pattern}, {self.dst_pattern})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def data_path(isa, src_pattern, dst_pattern):
    """The data path

    Args:
        isa (hardware.ISA): the data movement isa
        src_pattern (hardware.Pattern): the source memory pattern
        dst_pattern (hardware.Pattern): the dst memory pattern

    Returns:
        hardware.DataPath: the data path
    """
    return _ffi_api.DataPath(isa, src_pattern, dst_pattern)
