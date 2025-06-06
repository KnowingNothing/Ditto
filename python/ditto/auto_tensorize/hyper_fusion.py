import tvm
import tvm._ffi
from . import _ffi_api
from tvm.runtime import Object
from typing import List, Dict
from ditto import auto_compute
from ditto import hardware


@tvm._ffi.register_object("ditto.auto_tensorize.FusionChoice")
class FusionChoice(Object):
    """FusionChoice object"""

    def __str__(self) -> str:
        ret = f"FusionChoice(axis={self.ordered_iters}, pos={self.attach_pos})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def fusion_choice(
    first_op: tvm.te.tensor.ComputeOp,
    second_op: tvm.te.tensor.ComputeOp,
    ordered_iters: List[tvm.tir.IterVar],
    attach_pos: int,
):
    return _ffi_api.FusionChoice(first_op, second_op, ordered_iters, attach_pos)


@tvm._ffi.register_object("ditto.auto_tensorize.MatchInfo")
class MatchInfo(Object):
    """MatchInfo object"""

    def __str__(self) -> str:
        ret = f"MatchInfo(axis={self.axis}, intrin={self.intrin})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def match_info(axis: List[tvm.tir.IterVar], intrin: tvm.te.tensor_intrin.TensorIntrin, impl: str = ""):
    return _ffi_api.MatchInfo(axis, intrin, impl)


@tvm._ffi.register_object("ditto.auto_tensorize.TensorizeHyperFusionState")
class TensorizeHyperFusionState(Object):
    """TensorizeHyperFusionState object"""

    def __str__(self) -> str:
        ret = f"TensorizeHyperFusionState(layer={self.layer.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)

    def summary(self, verbose=False):
        ret = "TensorizeHyperFusionState Summary:\n"
        ret += "==================================\n"
        if verbose:
            ret += "Layer:\n"
            ret += "----------------------------------\n"
            ret += str(self.layer)
            ret += "----------------------------------\n"
        ret += f"First op: {self.first_op.name}\n"
        ret += f"Second op: {self.second_op.name}\n"
        if verbose:
            ret += "----------------------------------\n"
            ret += f"First op prologue:\n"
            ret += "----------------------------------\n"
            for v in self.first_op_prologue:
                tmp = "["
                for vv in v:
                    tmp += str(vv.name) + " "
                tmp += "]"
                ret += tmp + "\n"
            ret += "----------------------------------\n"
            ret += f"Second op prologue:\n"
            ret += "----------------------------------\n"
            for v in self.second_op_prologue:
                tmp = "["
                for vv in v:
                    tmp += str(vv.name) + " "
                tmp += "]"
                ret += tmp + "\n"
            ret += "----------------------------------\n"
            ret += "Inter Path:\n"
            tmp = "["
            for vv in self.inter_path:
                tmp += str(vv.name) + " "
            tmp += "]"
            ret += tmp + "\n"
            ret += "----------------------------------\n"
            ret += "Epilogue:\n"
            tmp = "["
            for vv in self.epilogue:
                tmp += str(vv.name) + " "
            tmp += "]"
            ret += tmp + "\n"
            ret += "----------------------------------\n"
            ret += f"Fused outer spatial iters: {self.fused_spatial_outer_iters}\n"
            ret += f"Fused outer reduce iters: {self.fused_reduce_outer_iters}\n"
            ret += "----------------------------------\n"
            ret += "Tensorize Information:\n"
            for k, v in self.tensorize_iters.items():
                ret += f"op={k.name}, axis={v}, intrin={self.tensorize_intrinsics[k]}\n"
        ret += "==================================\n"
        return ret


def tensorize_hyper_fusion_state(
    layer: auto_compute.Layer,
    fuse_choice: FusionChoice,
    match_info: Dict[tvm.te.tensor.ComputeOp, MatchInfo],
):
    return _ffi_api.TensorizeHyperFusionState(layer, fuse_choice, match_info)


@tvm._ffi.register_object("ditto.auto_tensorize.CUDATensorizeContext")
class CUDATensorizeContext(Object):
    """CUDATensorizeContext object"""


def cuda_tensorize_context(
    layer: auto_compute.Layer,
    state: TensorizeHyperFusionState,
    cuda_param: hardware.HardwareParam,
):
    return _ffi_api.CUDATensorizeContext(layer, state, cuda_param)


@tvm._ffi.register_object("ditto.auto_tensorize.CUDATensorizeParam")
class CUDATensorizeParam(Object):
    """CUDATensorizeParam object"""

    def __str__(self):
        return str(self.to_json())

    def __repr__(self) -> str:
        return str(self.to_json())
    
    def to_json(self):
        return {
            "warp_size": self.warp_size,
            "ty_size": self.ty_size,
            "tz_size": self.tz_size,
            "input_vector_len": self.input_vector_len,
            "serial_y": self.serial_y,
            "serial_z": self.serial_z,
            "block_rx": self.block_rx,
            "warp_rx": self.warp_rx,
            "block_ry": self.block_ry,
            "warp_ry": self.warp_ry,
            "unroll_steps": self.unroll_steps
        }


def cuda_tensorize_param(
    warp_size: int = 32,
    ty_size: int = 4,
    tz_size: int = 2,
    input_vector_len: int = 4,
    serial_y: int = 2,
    serial_z: int = 1,
    block_rx: int = 4,
    block_ry: int = 1,
    block_rz: int = 4,
    warp_rx: int = 1,
    warp_ry: int = 4,
    warp_rz: int = 1,
    unroll_steps: int = 1500,
):
    return _ffi_api.CUDATensorizeParam(
        warp_size,
        ty_size,
        tz_size,
        input_vector_len,
        serial_y,
        serial_z,
        block_rx,
        block_ry,
        block_rz,
        warp_rx,
        warp_ry,
        warp_rz,
        unroll_steps,
    )


def tensorize_cuda(
    layer: auto_compute.Layer,
    state: TensorizeHyperFusionState,
    cuda_param: hardware.HardwareParam,
    tensorize_param: CUDATensorizeParam,
):
    return _ffi_api.TensorizeCUDA(layer, state, cuda_param, tensorize_param)


@tvm._ffi.register_object("ditto.auto_tensorize.CPUTensorizeContext")
class CPUTensorizeContext(Object):
    """CPUTensorizeContext object"""


def cpu_tensorize_context(
    layer: auto_compute.Layer,
    state: TensorizeHyperFusionState,
    cpu_param: hardware.HardwareParam,
):
    return _ffi_api.CPUTensorizeContext(layer, state, cpu_param)


@tvm._ffi.register_object("ditto.auto_tensorize.CPUTensorizeParam")
class CPUTensorizeParam(Object):
    """CPUTensorizeParam object"""

    # def __str__(self):
    #     ret = "CPUTensorizeParam("
    #     ret += f"    first op tiling factor={self.firstOpTilingFactor}\n"
    #     ret += f"    first op loop order={self.firstOpLoopOrder}\n"
    #     ret += f"    second op tiling factor={self.secondOpTilingFactor}\n"
    #     ret += f"    second op loop order={self.secondOpLoopOrder}\n"
    #     ret += ")\n"
    #     return ret

    # def __repr__(self) -> str:
    #     return str(self)


def cpu_tensorize_param(
    warp_size: int = 32,
    ty_size: int = 4,
    tz_size: int = 2,
    input_vector_len: int = 4,
    serial_y: int = 2,
    serial_z: int = 1,
    block_rx: int = 4,
    block_ry: int = 1,
    block_rz: int = 4,
    warp_rx: int = 1,
    warp_ry: int = 4,
    warp_rz: int = 1,
    unroll_steps: int = 1500,
):
    return _ffi_api.CPUTensorizeParam(
        warp_size,
        ty_size,
        tz_size,
        input_vector_len,
        serial_y,
        serial_z,
        block_rx,
        block_ry,
        block_rz,
        warp_rx,
        warp_ry,
        warp_rz,
        unroll_steps,
    )


def tensorize_cpu(
    layer: auto_compute.Layer,
    state: TensorizeHyperFusionState,
    cpu_param: hardware.HardwareParam,
    tensorize_param: CPUTensorizeParam,
):
    return _ffi_api.TensorizeCPU(layer, state, cpu_param, tensorize_param)


def build_fusion_choice(
    sfs,
    hw_param: hardware.HardwareParam,
    dtype,
    simple_mode=-1,
):
    """Build the fusion choice.

    Parameters
    ----------
    sfs: the serial fusion state
    hw_param: the hardware parameters
    simple_mode:
        1: a random pick
        0: the heavy search method
        -1: the pruned search method
    Returns
    -------
    The fusion choice
    Example
    -------
    .. code-block:: python
    """
    return _ffi_api.buildFusionChoice(sfs, hw_param, dtype, simple_mode)


def build_cpu_tensorize_param(sfs, fusion_choice, hw_param, dtype):
    dtypeToBytes = {'float16':2, 'float32': 4, "float64": 8, "int32": 4, "int64": 8}
    if dtype not in dtypeToBytes:
        raise 
    bytePerEle = dtypeToBytes[dtype]
    return _ffi_api.buildCPUTensorizeParam(sfs, fusion_choice, hw_param, bytePerEle)
