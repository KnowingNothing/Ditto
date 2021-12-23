import pytest
from ditto import auto_tensorize as at


@pytest.mark.basic
def test_iter_var():
    """The iter var.
    """
    var1 = at.IterVar("name", 10)
    var2 = at.IterVar("name", 20)
    assert var1 == var2
    
    
@pytest.mark.basic
def test_iter_graph1():
    """The iter graph.
    """
    
    i1 = at.IterVar("i1", 1024)
    j1 = at.IterVar("j1", 1024)
    k1 = at.IterVar("k1", 1024)
    gemm1_iters = [i1, j1, k1]
    
    i2 = at.IterVar("i2", 1024)
    j2 = at.IterVar("j2", 1024)
    k2 = at.IterVar("k2", 1024)
    gemm2_iters = [i2, j2, k2]
    
    share_pairs = [
        (i1, i2),
        (j1, k2)
    ]
    
    g = at.IterGraph(gemm1_iters, gemm2_iters, share_pairs)
    g.setFirstOpTiling([7, 8, 9])
    g.setSecondOpTiling([10, 11, 12])
    g.fuseLoops(2)
    bounds = g.inferBound()
    with open("test_iter_graph_bounds.txt", "w") as fout:
        for iv, ext in bounds.items():
            print(iv, ext, file=fout)