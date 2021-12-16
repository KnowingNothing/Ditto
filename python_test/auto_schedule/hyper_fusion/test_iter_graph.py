import pytest
from ditto import auto_schedule as ash


@pytest.mark.basic
def test_iter_var():
    """[summary]
    """
    var1 = ash.IterVar("name", 10)
    var2 = ash.IterVar("name", 20)
    assert var1 == var2
    
    
@pytest.mark.basic
def test_iter_graph1():
    """[summary]
    """
    
    i1 = ash.IterVar("i1", 1024)
    j1 = ash.IterVar("j1", 1024)
    k1 = ash.IterVar("k1", 1024)
    gemm1_iters = [i1, j1, k1]
    
    i2 = ash.IterVar("i2", 1024)
    j2 = ash.IterVar("j2", 1024)
    k2 = ash.IterVar("k2", 1024)
    gemm2_iters = [i2, j2, k2]
    
    share_pairs = [
        (i1, i2),
        (j1, k2)
    ]
    
    g = ash.IterGraph(gemm1_iters, gemm2_iters, share_pairs)
    g.setFirstOpTiling([7, 8, 9])
    g.setSecondOpTiling([10, 11, 12])
    g.fuseLoops(2)
    bounds = g.inferBound()
    with open("test_iter_graph_bounds.txt", "w") as fout:
        for iv, ext in bounds.items():
            print(iv, ext, file=fout)