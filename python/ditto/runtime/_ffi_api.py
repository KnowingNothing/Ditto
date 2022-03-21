""" Register FFI APIs from C++ for the namespace ditto.runtime. """
import tvm._ffi


tvm._ffi._init_api("ditto.runtime", __name__)
