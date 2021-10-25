""" Register FFI APIs from C++ for the namespace ditto.autograd. """
import tvm._ffi


tvm._ffi._init_api("ditto.autograd", __name__)
