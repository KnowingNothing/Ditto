import tvm
from . import _ffi_api
from tvm.runtime import Object
from .iter_graph_ import IterGraph
from ditto.hardware.hw_param import HardwareParam


@tvm._ffi.register_object("ditto.auto_tensorize.FeatureLog")
class FeatureLog(Object):
    pass


@tvm._ffi.register_object("ditto.auto_tensorize.FusionResult")
class FusionResult(Object):
    def getLog(self):
        return _ffi_api.getLog(self)


def build_feature_log(ig: IterGraph, hp: HardwareParam):
    return _ffi_api.buildFeatureLog(ig, hp)
