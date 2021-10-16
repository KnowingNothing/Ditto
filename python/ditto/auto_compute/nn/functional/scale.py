import tvm


def channel_scale_nchw(x, alpha, beta):
    """
    ReLU
    ---
    args
    ---
    x: tvm.te.Tensor
        [batch, channel, height, width]
    alpha: tvm.te.Tensor 
        [channel]
    beta: tvm.te.Tensor
        [channel]
    Returns
    ---
    tvm.te.Tensor
    """
    return tvm.te.compute(
        x.shape,
        lambda n, c, h, w:
            x[n, c, h, w] * alpha[c] + beta[c],
        name="channel_scalue_nchw"
    )
