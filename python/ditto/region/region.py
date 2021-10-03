"""The computation region api of TVM."""
import tvm._ffi
from tvm._ffi.base import string_types

from tvm.runtime import Object, convert
from tvm.ir import container as _container
from tvm.tir import IterVar, Buffer

from . import _ffi_api

def create_region(op):
    """Create a region for single op

    Parameters
    ----------
    op : Operation
        The source expression.

    Returns
    -------
    reg : region.Region
        The created region.
    """
    return _ffi_api.CreateRegion(op)

@tvm._ffi.register_object
class Region(Object):
    """Region."""

    def split(self, parent, factor=None, nparts=None):
        """Split the region either by factor providing outer scope, or both

        Parameters
        ----------
        parent : IterVar
             The parent iter var.

        factor : Expr, optional
             The splitting factor

        nparts : Expr, optional
             The number of outer parts.

        Returns
        -------
        outer : IterVar
            The outer variable of iteration.

        inner : IterVar
            The inner variable of iteration.
        """
        if nparts is not None:
            if factor is not None:
                raise ValueError("Do not need to provide both outer and nparts")
            outer, inner = _ffi_api.RegionSplitByNParts(self, parent, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = _ffi_api.RegionSplitByFactor(self, parent, factor)
        return outer, inner

tvm._ffi._init_api("region", __name__)
