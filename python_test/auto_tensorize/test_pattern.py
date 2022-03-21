import pytest
import tvm
from ditto.auto_compute.nn import functional as F
from ditto import auto_compute as ac
from ditto import auto_tensorize as at


@pytest.mark.basic
def test_conv2d():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    Weight = tvm.te.placeholder([64, 16, 7, 7])
    conv2d = F.conv2d(Img, Weight, 2, 3, 1)
    assert at.get_op_pattern(conv2d.op) == ac.nn.pattern.PATTERN_CUBIC


@pytest.mark.basic
def test_linear():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    Weight = tvm.te.placeholder([112, 224])
    linear = F.linear(Img, Weight)
    assert at.get_op_pattern(linear.op) == ac.nn.pattern.PATTERN_CUBIC


@pytest.mark.basic
def test_relu():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    relu = F.ReLU(Img)
    print(relu.op.tag)
    print(ac.nn.pattern.PATTERN_LOCAL in relu.op.tag)
    assert at.get_op_pattern(relu.op) == ac.nn.pattern.PATTERN_LOCAL


@pytest.mark.basic
def test_add():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    Weight = tvm.te.placeholder([1, 16, 224, 224])
    res = F.add(Img, Weight)
    print(res.op.tag)
    print(ac.nn.pattern.PATTERN_LOCAL in res.op.tag)
    assert at.get_op_pattern(res.op) == ac.nn.pattern.PATTERN_LOCAL


@pytest.mark.basic
def test_transpose():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    res = F.transpose(Img, 1, 3)
    print(res.op.tag)
    print(ac.nn.pattern.PATTERN_SHUFFLE in res.op.tag)
    assert at.get_op_pattern(res.op) == ac.nn.pattern.PATTERN_SHUFFLE


@pytest.mark.basic
def test_im2col():
    Img = tvm.te.placeholder([1, 16, 224, 224])
    res = tvm.te.compute(
        [224 * 224 * 1, 16 * 9],
        lambda i, j: Img[
            i // (224 * 224),
            j // 9,
            i % (224 * 224) // 224 + j % 9 // 3,
            i % 224 + j % 3,
        ],
    )
    print(res.op.tag)
    print(ac.nn.pattern.PATTERN_SHUFFLE in res.op.tag)
    assert at.get_op_pattern(res.op) == ac.nn.pattern.PATTERN_SHUFFLE


if __name__ == "__main__":
    test_conv2d()
    test_linear()
    test_relu()
    test_add()
    test_transpose()
    test_im2col()
