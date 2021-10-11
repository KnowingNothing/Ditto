from typing import Union, Optional, List, Mapping
import warnings

import tvm.tir

from tvm.runtime import Module
from tvm.runtime import ndarray
from tvm.ir import container
from tvm.ir import CallingConv
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.target import codegen
from tvm.te import tensor
from tvm.te import schedule
from tvm.target import Target
from tvm.tir.buffer import Buffer
from tvm.tir.expr import Var
from ditto import hybrid

from . import _ffi_api as ffi

def get_binds(args, compact=False, binds=None):
    return tvm.driver.get_binds(args, compact, binds)


def schedule_to_module(
    sch: schedule.Schedule,
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
) -> IRModule:
    return tvm.driver.schedule_to_module(sch, args, name, binds)


def lower(
    inp: Union[schedule.Schedule, PrimFunc, IRModule, hybrid.HybridSchedule],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
    simple_mode: bool = False,
) -> IRModule:
    """Lowering step before build into target.

    See tvm.driver.lower for details
    """
    if isinstance(inp, IRModule):
        return tvm.driver.lower(inp, args, name, binds, simple_mode)
    if isinstance(inp, PrimFunc):
        return tvm.driver.lower(inp, args, name, binds, simple_mode)
    if isinstance(inp, schedule.Schedule):
        return tvm.driver.lower(inp, args, name, binds, simple_mode)
    if isinstance(inp, hybrid.HybridSchedule):
        return "This is a HybridSchedule"
    raise ValueError("Expected input to be an IRModule, PrimFunc, Schedule or HybridSchedule, but got, ", type(inp))


def _build_for_device(input_mod, target, target_host):
    return tvm.driver._build_for_device(input_mod, target, target_host)


def build(
    inputs: Union[schedule.Schedule, PrimFunc, IRModule, Mapping[str, IRModule], hybrid.HybridSchedule],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    target: Optional[Union[str, Target]] = None,
    target_host: Optional[Union[str, Target]] = None,
    name: Optional[str] = "default_function",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.
    
    See tvm.driver.build for details
    """
    if isinstance(inputs, schedule.Schedule):
        return tvm.driver.build(inputs, args, target, target_host, name, binds)
    elif isinstance(inputs, (list, tuple, container.Array)):
        return tvm.driver.build(inputs, args, target, target_host, name, binds)
    elif isinstance(inputs, (tvm.IRModule, PrimFunc)):
        return tvm.driver.build(inputs, args, target, target_host, name, binds)
    elif isinstance(inputs, (hybrid.HybridSchedule)):
        if args is None:
            raise ValueError("args must be given for build from hybrid_schedule")
        input_mod = lower(inputs, args, name=name, binds=binds)
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError(
            f"Inputs must be Schedule, IRModule, dict of target to IRModule or HybridSchedule, "
            f"but got {type(inputs)}."
        )

    if not isinstance(inputs, (dict, container.Map)):
        target = Target.current() if target is None else target
        target = target if target else "llvm"
        target_input_mod = {target: input_mod}
    else:
        target_input_mod = inputs

    for tar, mod in target_input_mod.items():
        if not isinstance(tar, (str, Target)):
            raise ValueError("The key of inputs must be str or " "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule," "or dict of str to IRModule.")

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )

    if not target_host:
        for tar, mod in target_input_mod.items():
            tar = Target(tar)
            device_type = ndarray.device(tar.kind.name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )

    mod_host_all = tvm.IRModule({})

    device_modules = []
    for tar, input_mod in target_input_mod.items():
        mod_host, mdev = _build_for_device(input_mod, tar, target_host)
        mod_host_all.update(mod_host)
        device_modules.append(mdev)

    # Generate a unified host module.
    rt_mod_host = codegen.build_module(mod_host_all, target_host)

    # Import all modules.
    for mdev in device_modules:
        if mdev:
            rt_mod_host.import_module(mdev)

    if not isinstance(target_host, Target):
        target_host = Target(target_host)
    if (
        target_host.attrs.get("runtime", tvm.runtime.String("c++")) == "c"
        and target_host.attrs.get("system-lib", 0) == 1
    ):
        if target_host.kind.name == "c":
            create_csource_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateCSourceCrtMetadataModule"
            )
            to_return = create_csource_crt_metadata_module([rt_mod_host], target_host)

        elif target_host.kind.name == "llvm":
            create_llvm_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateLLVMCrtMetadataModule"
            )
            to_return = create_llvm_crt_metadata_module([rt_mod_host], target_host)
    else:
        to_return = rt_mod_host

    return OperatorModule.from_module(to_return, ir_module_by_target=target_input_mod, name=name)


class OperatorModule(Module):
    """Wraps the Module returned by tvm.build() and captures additional outputs of that function."""

    @classmethod
    def from_module(cls, mod, **kwargs):
        # NOTE(areusch): It is generally unsafe to continue using `mod` from this point forward.
        # If an exception occurs in cls.__init__, handle will be deleted. For this reason,
        # set mod.handle to None.
        handle = mod.handle
        mod.handle = None
        return cls(handle, **kwargs)

    def __init__(self, handle, ir_module_by_target=None, name=None):
        super(OperatorModule, self).__init__(handle)
        self.ir_module_by_target = ir_module_by_target
        self.name = name
