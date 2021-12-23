""" Register FFI APIs from C++ for the namespace ditto.auto_tensorize. """
import tvm._ffi


tvm._ffi._init_api("ditto.auto_tensorize", __name__)
