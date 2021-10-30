import tvm
from tvm import testing
import ditto
import numpy as np

N1 = 6
N2 = 8
N3 = 10

A = tvm.te.placeholder([N1, N2], name="A")
B = tvm.te.placeholder([N2, N3], name="B")
k = tvm.te.reduce_axis((0, N2), name="k")
C = tvm.te.compute([N1, N3], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k), name="C")

s = ditto.hybrid.create_hybrid_schedule(C.op)

print(ditto.lower(s, [A, B, C], simple_mode=True))

i, j = s[C].op.axis
s[C].reorder(i, k, j)
s[C].bind(i, tvm.te.thread_axis("threadIdx.x"))
s[C].slice(j, mid=3, mode = "parallel")

s[C].display()

print(ditto.lower(s, [A, B, C], simple_mode=True))

tgt = tvm.target.Target(target="cuda", host="llvm")

f = ditto.build(s, [A, B, C], target=tgt, name="f")
print(f.imported_modules[0].get_source())

dev = tvm.device(tgt.kind.name, 0)

a = tvm.nd.array(np.random.uniform(size=[N1, N2]).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=[N2, N3]).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros([N1, N3], dtype=C.dtype), dev)
f(a, b, c)
testing.assert_allclose(c.numpy(), np.dot(a.numpy(), b.numpy()))
