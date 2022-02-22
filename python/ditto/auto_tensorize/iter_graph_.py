import tvm

from .state_ import SerialFusionState
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
    def synchronize(self):
        _ffi_api.synchronize(self)
    def set_schedule(self, fusionItem: FusionItem):
        _ffi_api.setSchedule(self, fusionItem)
    def analyse(self):
        return _ffi_api.getAnalyticalResult(self)
    def set_schedule(self, it: FusionItem):
        _ffi_api.setSchedule(self, it)
    def display(self):
        return _ffi_api.display(self)


def build_iter_graph(sfstate: SerialFusionState):
    return _ffi_api.build_iter_graph(sfstate)