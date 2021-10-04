import tvm
import ditto

A = tvm.te.placeholder([6, 6], name="A")
C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")

r = ditto.region.create_schedule(C.op)

# i, j = r[C].op.axis
# #ii, io = r[C].split(i, factor=1)
# C1, C2 = r.slice(C, i, factor=2)
