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
        return ffi.lower_hybrid_schedule(inp, args, name, binds, simple_mode)
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
        return tvm.driver.build(input_mod, args, target, target_host, name, binds)
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError(
            f"Inputs must be Schedule, IRModule, dict of target to IRModule or HybridSchedule, "
            f"but got {type(inputs)}."
        )
    return tvm.driver.build(inputs, args, target, target_host, name, binds)
