import tvm
import ditto

A = tvm.te.placeholder([6, 6], name="A")
C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")

s = ditto.region.create_region_schedule(C.op)
