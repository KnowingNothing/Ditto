import tvm


def batch_norm2d_nchw_v1(inputs, alpha, beta, epsilon=1e-5):
    """2D Batch Normalization for NCHW inputs

    Args:
    -----------------------------
    inputs  : tvm.te.Tensor
        shape [batch, channel, height, width]
    alpha   : tvm.te.Tensor
        shape [channel]
    beta    : tvm.te.Tensor
        shape [channel]
    epsilon : float
        optional
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.Tensor
        shape [batch, channel, height, width]
    -----------------------------
    """
    N, C, H, W = inputs.shape
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    assert (len(alpha.shape) == 1) and (alpha.shape[0] == C)
    assert (len(beta.shape) == 1) and (beta.shape[0] == C)

    rn1 = tvm.te.reduce_axis([0, N], name="rn1")
    rh1 = tvm.te.reduce_axis([0, H], name="rh1")
    rw1 = tvm.te.reduce_axis([0, W], name="rw1")

    mean = tvm.te.compute(
        [C],
        lambda c: tvm.te.sum(
            inputs[rn1, c, rh1, rw1] / (N * H * W), axis=[rn1, rh1, rw1]
        ),
        name="bn_mean",
    )

    rn2 = tvm.te.reduce_axis([0, N], name="rn2")
    rh2 = tvm.te.reduce_axis([0, H], name="rh2")
    rw2 = tvm.te.reduce_axis([0, W], name="rw2")

    square = tvm.te.compute(
        [C],
        lambda c: tvm.te.sum(
            (inputs[rn2, c, rh2, rw2] * inputs[rn2, c, rh2, rw2]) / (N * H * W),
            axis=[rn2, rh2, rw2],
        ),
        name="bn_square",
    )

    variance = tvm.te.compute(
        [C], lambda c: square[c] - mean[c] * mean[c], name="variance"
    )

    bn = tvm.te.compute(
        [N, C, H, W],
        lambda n, c, h, w: (inputs[n, c, h, w] - mean[c])
        / tvm.te.sqrt(variance[c] + epsilon)
        * alpha[c]
        + beta[c],
        name="bn",
    )

    return bn


def batch_norm2d_nchw_v2(inputs, mean, variance, alpha, beta, epsilon=1e-5):
    """2D Batch Normalization for NCHW inputs

    Args:
    -----------------------------
    inputs  : tvm.te.Tensor
        shape [batch, channel, height, width]
    mean    : tvm.te.Tensor
        shape [channel]
    variance: tvm.te.Tensor
        shape [channel]
    alpha   : tvm.te.Tensor
        shape [channel]
    beta    : tvm.te.Tensor
        shape [channel]
    epsilon : float
        optional
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.Tensor
        shape [batch, channel, height, width]
    -----------------------------
    """
    N, C, H, W = inputs.shape
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    assert (len(mean.shape) == 1) and (mean.shape[0] == C)
    assert (len(variance.shape) == 1) and (variance.shape[0] == C)
    assert (len(alpha.shape) == 1) and (alpha.shape[0] == C)
    assert (len(beta.shape) == 1) and (beta.shape[0] == C)

    bn = tvm.te.compute(
        [N, C, H, W],
        lambda n, c, h, w: (inputs[n, c, h, w] - mean[c])
        / tvm.te.sqrt(variance[c] + epsilon)
        * alpha[c]
        + beta[c],
        name="bn",
    )

    return bn
