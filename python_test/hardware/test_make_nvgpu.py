import pytest
import tvm
from ditto import hardware as hw


@pytest.mark.basic
def test_make_v100():
    """Make a V100 device
    """
    v100 = hw.query_hw("gpu.cuda.v100")
    print(v100.summary())
    

if __name__ == "__main__":
    test_make_v100()