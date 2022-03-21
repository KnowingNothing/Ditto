import tvm
from .padding import zero_pad2d


def avgpool2d_nchw(inputs, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, padding=0):
    """Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, channel, height, width]
    Stride, padding are also supported

    out_height = (height + 2*padding - kernel_h) // stride_h + 1

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, out_height, out_width]
    """
    padding = (
        (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    )
    padded_inputs = zero_pad2d(inputs, padding=padding)

    batch, channel, h_in, w_in = padded_inputs.shape

    h_out = (h_in - kernel_h) // stride_h + 1
    w_out = (w_in - kernel_w) // stride_w + 1

    r = tvm.te.reduce_axis([0, kernel_h], name="r")
    s = tvm.te.reduce_axis([0, kernel_w], name="s")

    return tvm.te.compute(
        [batch, channel, h_out, w_out],
        lambda n, c, i, j: tvm.te.sum(
            padded_inputs[n, c, i * stride_h + r, j * stride_w + s]
            / (kernel_h * kernel_w),
            axis=[r, s],
        ),
        name="avgpool2d_nchw",
    )


def global_avgpool2d_nchw(inputs, keep_dim=True):
    """Global Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, channel, height, width]

        keep_dim: bool

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, 1, 1] if keep dim is True
        else [batch, channel]
    """
    N, C, H, W = inputs.shape
    if keep_dim:
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return tvm.te.compute(
            [N, C, 1, 1],
            lambda n, c, i, j: tvm.te.sum(inputs[n, c, h, w] / (H * W), axis=[h, w]),
            name="global_avgpool2d_nchw_keep",
        )
    else:
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return tvm.te.compute(
            [N, C],
            lambda n, c: tvm.te.sum(inputs[n, c, h, w] / (H * W), axis=[h, w]),
            name="global_avgpool2d_nchw",
        )
