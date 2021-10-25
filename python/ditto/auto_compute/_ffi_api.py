""" Register FFI APIs from C++ for the namespace ditto.auto_compute. """
import tvm._ffi


tvm._ffi._init_api("ditto.auto_compute", __name__)
