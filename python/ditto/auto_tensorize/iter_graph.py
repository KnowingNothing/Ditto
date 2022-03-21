import tvm

from .state import SerialFusionState
from . import _ffi_api
from tvm.runtime import Object
from typing import *
from .item import FusionItem


@tvm._ffi.register_object("ditto.auto_tensorize.IterGraph")
class IterGraph(Object):
    def set_attach(self, attach_pos: int):
        _ffi_api.setAttach(self, attach_pos)

    def set_first_op_tiling(self, factors: Sequence[int]):
        _ffi_api.setFirstOpTiling(self, factors)

    def set_second_op_tiling(self, factors: Sequence[int]):
        _ffi_api.setSecondOpTiling(self, factors)

    def set_first_op_permute(self, factors: Sequence[int]):
        _ffi_api.setFirstOpPermute(self, factors)

    def set_second_op_permute(self, factors: Sequence[int]):
        _ffi_api.setSecondOpPermute(self, factors)

    def apply_all(self):
        _ffi_api.applyAll(self)

    def set_fusion(self, fusionItem: FusionItem):
        _ffi_api.setFusion(self, fusionItem)

    def analyse(self, hw_param, bytePerEle, writeThrough):
        """Get the analysis result

        Parameters
        ----------
        hw_param: hardware params

        bytePerEle: int
            the byte taken by single element; e.x. float32 --> 4
        writeThrough: bool
            whether write E reside in the shared memory
        Returns
        -------
        out : FusionResult
            The analysis result
        """

        return _ffi_api.getAnalyticalResult(self, hw_param, bytePerEle, writeThrough)

    def display(self):
        return _ffi_api.display(self)


def build_iter_graph(
    sfstate: SerialFusionState, tensorizeItervars: List[tvm.tir.IterVar], path: str = "", tensorWeight: List[float] = [1.0, 1.0] 
):
    return _ffi_api.build_iter_graph(sfstate, tensorizeItervars, tensorWeight, path)
