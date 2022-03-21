from .. import _ffi_api
import tvm._ffi
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.auto_compute.Pattern")
class Pattern(Object):
    """Pattern object"""

    def to_json(self):
        ret = {}
        ret["tensor_ids"] = [x.value for x in self.tensor_ids]
        ret["iter_ids_array"] = [[y.value for y in x] for x in self.iter_ids_array]
        return ret

    @staticmethod
    def from_json(obj):
        tensor_ids = obj["tensor_ids"]
        iter_ids_array = obj["iter_ids_array"]
        return _ffi_api.MakePattern(tensor_ids, iter_ids_array)

    @staticmethod
    def empty_pattern():
        return Pattern.from_json({"tensor_ids": [], "iter_ids_array": []})
