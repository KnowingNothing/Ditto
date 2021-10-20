import tvm


def add(A, B):
    shape = [int(x) for x in A.shape]
    shape_B = [int(x) for x in B.shape]
    assert shape == shape_B
    return tvm.te.compute(
        shape,
        lambda *idx: A(*idx) + B(*idx),
        name="add"
    )
