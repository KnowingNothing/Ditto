""" Register FFI APIs from C++ for the namespace tvm.nast. """
import tvm._ffi


tvm._ffi._init_api("ditto", __name__)
