import tvm
from tvm import te
import numpy as np


N = 128

A = te.placeholder([N, N], name="A", dtype="float32")
A_state = te.placeholder([N, N, N], name="A_state", dtype="float32")
R_state = te.placeholder([N, N, N], name="R_state", dtype="float32")

A_init = te.compute([1, N, N], lambda _, i, j: A[i, j])
R_init = te.compute(
    [1, N, N],
    lambda _, i, j: tvm.tir.if_then_else(
        0 == i,
        tvm.tir.if_then_else(
            j == 0,
            te.sqrt(A_state[0, i, i]),
            A_state[0, i, j] / te.sqrt(A_state[0, 0, 0]),
        ),
        0.0,
    ),
)
A_update = te.compute(
    [N, N, N],
    lambda t, i, j: tvm.tir.if_then_else(
        tvm.tir.all(i >= t, j >= t),
        A_state[t - 1, i, j]
        - R_state[t - 1, t - 1, i] * R_state[t - 1, t - 1, j]
        + 1e-10,
        A_state[t - 1, i, j],
    ),
)
R_update = te.compute(
    [N, N, N],
    lambda t, i, j: tvm.tir.if_then_else(
        tvm.tir.all(t == i, t == j),
        te.sqrt(A_state[t, i, i]),
        tvm.tir.if_then_else(
            tvm.tir.all(i == t, j > t),
            A_state[t, i, j] / te.sqrt(A_state[t, t, t]),
            R_state[t - 1, i, j],
        ),
    ),
)

A_scan, R_scan = tvm.te.scan(
    [A_init, R_init], [A_update, R_update], [A_state, R_state], inputs=[A]
)

s = te.create_schedule(R_scan.op)

print(R_scan.op.scan_axis)
print(R_scan.op.init)
print(R_scan.op.update)
print(R_scan.op.state_placeholder)
print(dir(R_scan.op))

factor_x = 2
factor_y = 16

block_x = te.thread_axis((0, N//factor_x), "blockIdx.x")
block_y = te.thread_axis((0, N//factor_y), "blockIdx.y")
thread_x = te.thread_axis((0, factor_x), "threadIdx.x")
thread_y = te.thread_axis((0, factor_y), "threadIdx.y")
s[R_scan].env_threads([block_x, block_y, thread_y, thread_x])

t, i, j = s[R_init].op.axis
io, ii = s[R_init].split(i, factor=factor_y)
jo, ji = s[R_init].split(j, factor=factor_x)
s[R_init].reorder(io, jo, ii, ji)
s[R_init].bind(io, block_y)
s[R_init].bind(jo, block_x)
s[R_init].bind(ii, thread_y)
s[R_init].bind(ji, thread_x)

t, i, j = s[A_init].op.axis
io, ii = s[A_init].split(i, factor=factor_y)
jo, ji = s[A_init].split(j, factor=factor_x)
s[A_init].reorder(io, jo, ii, ji)
s[A_init].bind(io, block_y)
s[A_init].bind(jo, block_x)
s[A_init].bind(ii, thread_y)
s[A_init].bind(ji, thread_x)

t, i, j = s[R_update].op.axis
io, ii = s[R_update].split(i, factor=factor_y)
jo, ji = s[R_update].split(j, factor=factor_x)
s[R_update].reorder(io, jo, ii, ji)
s[R_update].bind(io, block_y)
s[R_update].bind(jo, block_x)
s[R_update].bind(ii, thread_y)
s[R_update].bind(ji, thread_x)

t, i, j = s[A_update].op.axis
io, ii = s[A_update].split(i, factor=factor_y)
jo, ji = s[A_update].split(j, factor=factor_x)
s[A_update].reorder(io, jo, ii, ji)
s[A_update].reorder(io, jo, ii, ji)
s[A_update].bind(io, block_y)
s[A_update].bind(jo, block_x)
s[A_update].bind(ii, thread_y)
s[A_update].bind(ji, thread_x)

print(tvm.lower(s, [A, R_scan], simple_mode=True))

with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 128,
            },
            "tir.detect_global_barrier": True,
        }
    ):
    func = tvm.build(s, [A, R_scan], target="cuda")
# func = tvm.build(s, [A, R_scan], target="llvm")
print("Number of device kernels:")
print(len(func.imported_modules))
with open("trace_cholesky_kernel.log", "w") as fout:
    print(func.imported_modules[0].get_source(), file=fout)

A_np = np.random.uniform(10, 15, [N, N]).astype("float32")
for j in range(N):
    for i in range(j + 1, N):
        A_np[i, j] = 0.0

print("Triangular matrix:")
print(A_np)

A_np = np.matmul(A_np.T, A_np)
print("Symmetric matrix:")
print(A_np)

R_np = np.random.uniform(-10, 10, [N, N, N]).astype("float32")
dev = tvm.cuda(0)
# dev = tvm.cpu(0)
A_tvm = tvm.nd.array(A_np, dev)
R_tvm = tvm.nd.array(R_np, dev)

func(A_tvm, R_tvm)


def compute_golden(X):
    Y = np.copy(X)
    Z = np.random.uniform(-10, 10, [N, N, N]).astype("float32") * 0.0
    for i in range(N):
        for j in range(N):
            Y[i, j] = X[i, j]

            if (i == 0) and (j == 0):
                Z[0, i, j] = np.sqrt(X[i, j])
            elif (i == 0) and (j > 0):
                Z[0, i, j] = X[i, j] / np.sqrt(X[0, 0])
            else:
                Z[0, i, j] = 0.0
    for t in range(1, N):
        for i in range(t, N):
            for j in range(t, N):
                Y[i, j] = Y[i, j] - Z[t - 1, t - 1, i] * Z[t - 1, t - 1, j]
        for i in range(N):
            for j in range(N):
                Z[t, i, j] = Z[t - 1, i, j]
        for j in range(t, N):
            if j == t:
                Z[t, t, t] = np.sqrt(Y[t, t])
            elif j > t:
                Z[t, t, j] = Y[t, j] / np.sqrt(Y[t, t])
    return Z


golden_np = compute_golden(A_np)

print("Golden Value")
print(golden_np[-1, :, :])

from tvm import testing

testing.assert_allclose(golden_np, R_tvm.numpy(), atol=1e-3, rtol=1e-3)
