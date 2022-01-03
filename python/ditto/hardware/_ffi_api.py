""" Register FFI APIs from C++ for the namespace ditto.hardware. """
import tvm._ffi


tvm._ffi._init_api("ditto.hardware", __name__)
