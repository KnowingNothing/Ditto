"""The computation region api of TVM."""
import tvm._ffi
from tvm._ffi.base import string_types

from tvm.runtime import Object, convert
from tvm.ir import container as _container
from tvm.tir import IterVar, Buffer

from . import _ffi_api

def create_region(ops):
    """Create a region for list of ops

    Parameters
    ----------
    ops : list of Operations
        The source expression.

    Returns
    -------
    reg : region.Region
        The created region.
    """
    if not isinstance(ops, (list, _container.Array)):
        ops = [ops]
    return _ffi_api.CreateRegion(ops)

@tvm._ffi.register_object
class Region(Object):
    """Region."""

tvm._ffi._init_api("region", __name__)
