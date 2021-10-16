import tvm
import ditto

A = tvm.te.placeholder([6, 6], name="A")
C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")

s = ditto.hybrid.create_hybrid_schedule(C.op)

print(ditto.lower(s, [A, C], simple_mode=True))
print(ditto.lower(s, [A, C], simple_mode=True))
print(ditto.lower(s, [A, C], simple_mode=True))

i, j = s[C].op.axis

print('end')