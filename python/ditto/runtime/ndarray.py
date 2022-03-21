import tvm
import tvm._ffi
import numpy as np


@tvm._ffi.register_func("runtime.ndarray.random")
def ndarray_random(shape, low, high, dtype, device):
    low = low.value if isinstance(low, tvm.tir.FloatImm) else low
    high = high.value if isinstance(high, tvm.tir.FloatImm) else high
    ary = np.random.uniform(low, high, [int(x) for x in shape]).astype(dtype)
    return tvm.nd.array(ary, device)
