import tvm
import tvm._ffi
from . import _ffi_api
from tvm.runtime import Object
from typing import List, Dict


@tvm._ffi.register_object("ditto.auto_tensorize.PackedIntrinsic")
class PackedIntrinsic(Object):
    """PackedIntrinsic object"""

    def __str__(self):
        ret = f"PackedIntrinsic(\n"
        ret += f"    load={self.load_intrinsics}\n"
        ret += f"    compute={self.compute_intrinsic}\n"
        ret += f"    store={self.store_intrinsic})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def packed_intrinsic(
    loads: List[tvm.te.tensor_intrin.TensorIntrin],
    compute: tvm.te.tensor_intrin.TensorIntrin,
    store: tvm.te.tensor_intrin.TensorIntrin,
):
    return _ffi_api.PackedIntrinsic(loads, compute, store)
