import tvm
import tvm._ffi
from . import _ffi_api
from tvm.runtime import Object
from typing import List, Dict
from ditto import auto_compute


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


def match_info(axis: List[tvm.tir.IterVar], intrin: tvm.te.tensor_intrin.TensorIntrin):
    return _ffi_api.MatchInfo(axis, intrin)


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
