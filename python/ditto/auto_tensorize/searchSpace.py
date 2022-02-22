import tvm
from . import _ffi_api
from tvm.runtime import Object
from typing import Sequence

@tvm.register_object("ditto.auto_tensorize.SearchSpace")
class SearchSpace(Object):
    pass

@tvm.register_object("ditto.auto_tensorize.FusionSpace")
class FusionSpace(SearchSpace):
    def set_first_op_tiling_mandatories(self, factors: Sequence[Sequence[int]]):
        _ffi_api.setFirstOpTilingMandatory(self, factors)
    def set_second_op_tiling_mandatories(self, factors: Sequence[Sequence[int]]):
        _ffi_api.setSecondOpTilingMandatory(self, factors)