""" Register FFI APIs from C++ for the namespace ditto.auto_schedule. """
import tvm._ffi


tvm._ffi._init_api("ditto.auto_schedule", __name__)
