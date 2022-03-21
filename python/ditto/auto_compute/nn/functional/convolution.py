import tvm
from tvm.topi.nn import conv2d
from .padding import zero_pad2d


def conv2d_with_bias_nchw(
    inputs, weight, bias=None, stride=1, padding=0, dilation=1, out_dtype="float32"
):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape

    assert in_channel == channel_per_group

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (
        (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    )
    dilation = (
        (dilation, dilation)
        if isinstance(dilation, (int, tvm.tir.IntImm))
        else dilation
    )
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w)
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    conv_out = tvm.te.compute(
        conv_out_shape,
        lambda b, c, h, w: tvm.te.sum(
            (
                padded[
                    b,
                    rc,
                    h * stride[0] + rh * dilation[0],
                    w * stride[1] + rw * dilation[1],
                ]
                * weight[c, rc, rh, rw]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
        name="conv2d_with_bias_nchw",
    )

    if bias is not None:
        return tvm.te.compute(
            conv_out_shape,
            lambda b, c, h, w: conv_out[b, c, h, w] + bias[c],
            name="conv2d_with_bias_nchw_bias",
        )
    return conv_out


def conv2d_with_group_nchw(
    inputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out_dtype="float32",
):
    """Convolution 2d NCHW layout, grouped

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert channel_per_group * groups == in_channel, "%d vs. %d" % (
        channel_per_group * groups,
        in_channel,
    )
    out_channel_per_group = out_channel // groups
    assert out_channel_per_group * groups == out_channel

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (
        (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    )
    dilation = (
        (dilation, dilation)
        if isinstance(dilation, (int, tvm.tir.IntImm))
        else dilation
    )
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)

    data_reshape = tvm.te.compute(
        [
            batch_size,
            groups,
            channel_per_group,
            in_h + 2 * padding[0],
            in_w + 2 * padding[1],
        ],
        lambda n, g, c, h, w: padded[n, g * channel_per_group + c, h, w],
        name="reshape_data",
    )

    kernel_reshape = tvm.te.compute(
        [groups, out_channel_per_group, channel_per_group, k_h, k_w],
        lambda g, k, c, r, s: weight[g * out_channel_per_group + k, c, r, s],
        name="reshape_kernel",
    )

    conv_out_shape = (batch_size, groups, out_channel_per_group, out_h, out_w)

    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    conv_out = tvm.te.compute(
        conv_out_shape,
        lambda b, g, k, h, w: tvm.te.sum(
            (
                data_reshape[
                    b,
                    g,
                    rc,
                    h * stride[0] + rh * dilation[0],
                    w * stride[1] + rw * dilation[1],
                ]
                * kernel_reshape[g, k, rc, rh, rw]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
        name="conv2d_nchw_grouped",
    )

    output = tvm.te.compute(
        [batch_size, out_channel, out_h, out_w],
        lambda b, c, h, w: conv_out[
            b, c // out_channel_per_group, c % out_channel_per_group, h, w
        ],
        name="reshape_output",
    )

    if bias is not None:
        return tvm.te.compute(
            [batch_size, out_channel, out_h, out_w],
            lambda b, c, h, w: output[b, c, h, w] + bias[c],
            name="conv2d_nchw_bias",
        )
    return output


def conv2d_capsule_nchw(
    inputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    num_caps=8,
    out_dtype="float32",
):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width, num_caps]
    bias    : (optional:None) GraphNode
        shape [out_channel, num_caps]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    num_caps : (optional:8) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width, num_caps]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w, num_caps_ = weight.shape
    assert channel_per_group == in_channel
    assert num_caps_ == num_caps

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (
        (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    )
    dilation = (
        (dilation, dilation)
        if isinstance(dilation, (int, tvm.tir.IntImm))
        else dilation
    )
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w, num_caps)

    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    conv_out = tvm.te.compute(
        conv_out_shape,
        lambda b, c, h, w, s: tvm.te.sum(
            (
                padded[
                    b,
                    rc,
                    h * stride[0] + rh * dilation[0],
                    w * stride[1] + rw * dilation[1],
                ]
                * weight[c, rc, rh, rw, s]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
        name="conv2d_capsule_nchw",
    )

    if bias is not None:
        return tvm.te.compute(
            conv_out_shape,
            lambda b, c, h, w, s: conv_out[b, c, h, w, s] + bias[c, s],
            name="conv2d_capsule_nchw_bias",
        )

    return conv_out
