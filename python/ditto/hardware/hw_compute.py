import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.HardwareCompute")
class HardwareCompute(Object):
    """HardwareCompute object"""

    def __str__(self) -> str:
        ret = f"HardwareCompute({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.HardwareUnit")
class HardwareUnit(HardwareCompute):
    """HardwareUnit object"""

    def __str__(self) -> str:
        ret = f"HardwareUnit({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_unit(supported_isa, name="hw_unit"):
    """Hardware unit

    Args:
        supported_isa (Map[str, hardware.ISA]): the supported virtual isa
        name (str, optional): the name of this unit. Defaults to "hw_unit".

    Returns:
        hardware.HardwareUnit: the hardware unit
    """
    return _ffi_api.HardwareUnit(name, supported_isa)


@tvm._ffi.register_object("ditto.hardware.HardwareProcessor")
class HardwareProcessor(HardwareCompute):
    """HardwareProcessor object"""

    def __str__(self) -> str:
        ret = f"HardwareProcessor({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.HeteroProcessor")
class HeteroProcessor(HardwareProcessor):
    """Hardware HeteroProcessor object"""

    def __str__(self) -> str:
        ret = f"HeteroProcessor({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_heteroprocessor(units, local_mems, topology, name="hw_heteroprocessor"):
    """Hardware heterogeneous processor

    Args:
        units (List[hardware.HardwareUnit]): the units in this processor
        local_mems (List[hardware.LocalMemory]): the local memory in this processor
        topology (Map[hardware.Hardware, Map[hardware.LocalMemory, hardware.HardwarePath]]): the topology of this processor
        name (str, optional): the name of this processor. Defaults to "hw_heteroprocessor".

    Returns:
        hardware.HeteroProcessor: the hardware processor
    """
    return _ffi_api.HeteroProcessor(name, units, local_mems, topology)


@tvm._ffi.register_object("ditto.hardware.HardwareGroup")
class HardwareGroup(HardwareCompute):
    """HardwareGroup object"""

    def __str__(self) -> str:
        ret = f"HardwareGroup({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.HomoGroup")
class HomoGroup(HardwareGroup):
    """Hardware HomoGroup object"""

    def __str__(self) -> str:
        ret = f"HomoGroup({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_homogroup(processor, shared_mem, block_x, block_y=1, block_z=1, name="hw_homogroup"):
    """Hardware homogeneous group

    Args:
        processor (hardware.Processor): the processor in this group
        shared_mem (hardware.SharedMemory): the shared memory in this group
        block_x (int): the x dim
        block_y (int, optional): the y dim. Defaults to 1.
        block_z (int, optional): the z dim. Defaults to 1.
        name (str, optional): the name of this group. Defaults to "hw_homogroup".

    Returns:
        hardware.HomoGroup: the hardware homogeneous group
    """
    return _ffi_api.HomoGroup(name, processor, shared_mem, block_x, block_y, block_z)


@tvm._ffi.register_object("ditto.hardware.HardwareDevice")
class HardwareDevice(HardwareCompute):
    """HardwareDevice object"""

    def __str__(self) -> str:
        ret = f"HardwareDevice({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def hw_device(group, global_mem, grid_x, grid_y=1, grid_z=1, name="hw_device"):
    """Make a hardware device

    Args:
        group (hardware.HardwareGroup): the group that composes the device
        global_mem (hardware.GlobalMemory): the global memory in the device
        grid_x (int): the x dim
        grid_y (int, optional): the y dim. Defaults to 1.
        grid_z (int, optional): the z dim. Defaults to 1.
        name (str, optional): the name of the device. Defaults to "hw_device".

    Returns:
        hardware.HardwareDevice: the hardware device
    """
    return _ffi_api.HardwareDevice(name, group, global_mem, grid_x, grid_y, grid_z)
