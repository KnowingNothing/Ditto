import tvm
from . import _ffi_api
from tvm.runtime import Object
import ditto.auto_compute as ac

@tvm._ffi.register_object("ditto.auto_tensorize.SerialFusionState")
class SerialFusionState(Object):
    pass

def build_serial_fusion_state(layer: ac.Layer):
    return _ffi_api.build_serial_fusion_state(layer)