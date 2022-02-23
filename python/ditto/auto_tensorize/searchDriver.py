import tvm
from . import _ffi_api
from .iter_graph_ import IterGraph
from tvm.runtime import Object 
from typing import Sequence
from .item import Item

@tvm.register_object('ditto.auto_tensorize.SearchDriver')
class SearchDriver(Object):
    def search(self):
        """
        Do the search

        Parameters
        -----

        Returns
        -----
        item: List[Result]
            the best item
        """
        return _ffi_api.search(self)
    def eval(self, it: Item):
        """
        Evaluate the given item

        Parameters
        -----
        it: Item
            the item to evaluate

        Returns
        -----
        evaluate result: List[Result]
        """
        return _ffi_api.eval(self, it)
    def get_fusion_space(self):
        return _ffi_api.getFusionSpace(self)

def build_search_driver(ig: IterGraph, evals: Sequence[str], searcherType: str, hw_param, dtype):
    return _ffi_api.buildSearchDriver(ig, evals, searcherType, hw_param, dtype)

