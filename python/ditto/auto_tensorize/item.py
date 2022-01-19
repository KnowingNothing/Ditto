import tvm
from . import _ffi_api
from tvm import Object 
from typing import Sequence


@tvm.register_object("ditto.auto_tensorize.Item")
class Item(Object):
    pass 

@tvm.register_object("ditto.auto_tensorize.FusionItem")
class FusionItem(Item):
    pass

def build_fusion_item(  firstOpTiling: Sequence[int], \
                        secondOpTiling: Sequence[int], \
                        firstOpPermute: Sequence[int], \
                        secondOpPermute: Sequence[int], \
                        attachPos: int):
    return _ffi_api.buildFusionItem(firstOpTiling, secondOpTiling,\
        firstOpPermute, secondOpPermute, attachPos)

@tvm.register_object("ditto.auto_tensorize.Result")
class Result(Object):
    pass 