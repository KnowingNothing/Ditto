import pytest
import tvm
from ditto import auto_compute as ac
from ditto import auto_schedule as ash
import time


def GemmReLUGemm(M, N, K, L):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, L], name="B", dtype="float32")
    E = tvm.te.placeholder([L, N], name="E", dtype="float32")
    k = tvm.te.reduce_axis([0, K], "rk")
    C = tvm.te.compute(
        [M, L],
        lambda i, j:
            tvm.te.sum(A[i, k] * B[k, j], axis=k),
        name="C"
    )
    D = tvm.te.compute(
        [M, L],
        lambda i, j:
            tvm.tir.if_then_else(
                C[i, j] > 0,
                C[i, j],
                0.0
        ),
        name="D"
    )
    l = tvm.te.reduce_axis([0, L], "rl")
    F = tvm.te.compute(
        [M, N],
        lambda i, j:
            tvm.te.sum(
                D[i, l] * E[l, j], axis=l
        ),
        name="F"
    )
    return [A, B, E], [F]


@pytest.mark.basic
def test_space_size():
    """The search space size.
    """
    M = 512
    N = 64
    K = 64
    L = 512
    ins, outs = GemmReLUGemm(M, N, K, L)
    A, B, E = ins
    F, = outs
    layer = ac.layer(F.op, inputs=[A, B, E])
    hyper_state = ash.build_hyper_state(layer)
    iter_graph = hyper_state.build_iter_graph()
    fuse_tile_space = ash.FusionTileSpace(iter_graph)
    print(f"Space size is {len(fuse_tile_space)}.")
    for k, v in fuse_tile_space.subspaces.items():
        print(f"Subspace {k} size is {len(v)}")
    counter = 0
    for item in fuse_tile_space.all_items():
        assert item
        counter += 1
    assert counter == len(fuse_tile_space)


@pytest.mark.basic
def test_space_item():
    """The search space size.
    """
    M = 512
    N = 64
    K = 64
    L = 512
    ins, outs = GemmReLUGemm(M, N, K, L)
    A, B, E = ins
    F, = outs
    layer = ac.layer(F.op, inputs=[A, B, E])
    hyper_state = ash.build_hyper_state(layer)
    iter_graph = hyper_state.build_iter_graph()
    fuse_tile_space = ash.FusionTileSpace(iter_graph)
    beg = time.time()
    for iter_graph in fuse_tile_space.all_iter_graphs():
        bounds = iter_graph.inferBound()
    end = time.time()
    print(f"Use time {end - beg}s to generate all iter_graphs.")


if __name__ == "__main__":
    test_space_size()
    test_space_item()
