import tvm
from . import _ffi_api
from tvm.runtime import Object
from typing import Sequence


@tvm.register_object("ditto.auto_tensorize.SearchSpace")
class SearchSpace(Object):
    pass


@tvm.register_object("ditto.auto_tensorize.FusionSpace")
class FusionSpace(SearchSpace):
    def set_first_op_tiling_mandatory(self, factors: Sequence[int]):
        """
        factors:
            user given tiling factor
            -1 indicates the axis is not set, other means the mandatary value
        """
        _ffi_api.setFirstOpTilingMandatory(self, factors)

    def set_second_op_tiling_mandatory(self, factors: Sequence[int]):
        _ffi_api.setSecondOpTilingMandatory(self, factors)

    def set_first_op_permute_mandatory(self, factors: Sequence[Sequence[int]]):
        _ffi_api.setFirstOpPermuteMandatory(self, factors)

    def set_second_op_permute_mandatory(self, factors: Sequence[Sequence[int]]):
        _ffi_api.setSecondOpPermuteMandatory(self, factors)

    def set_attach_mandatory(self, factors: Sequence[int]):
        _ffi_api.setAttachMandatory(self, factors)
