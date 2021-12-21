""" Register FFI APIs from C++ for the namespace ditto.utils. """
import tvm._ffi


tvm._ffi._init_api("ditto.utils", __name__)
