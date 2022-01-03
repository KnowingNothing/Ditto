import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.Hardware")
class Hardware(Object):
    """Hardware object"""

    def __str__(self) -> str:
        ret = f"Hardware({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)
