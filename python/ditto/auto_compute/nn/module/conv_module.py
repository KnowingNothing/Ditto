import tvm
from .module import Module, Parameter
from ..functional import (
    conv2d,
    conv2d_with_bias_nchw,
    conv2d_with_group_nchw,
    conv2d_capsule_nchw,
)
from ...graph import LayerTensor, layer


class Conv2d(Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        bias=False,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        dtype="float32",
        out_dtype="float32",
        layout="NCHW",
    ):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
        stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, (list, tuple)) and len(stride) == 2
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, (tuple, list)) and len(padding) == 2
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
        assert isinstance(groups, int)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.layout = layout

        self.weight = Parameter(
            (out_channel, in_channel // groups, *kernel_size),
            dtype=dtype,
            name="conv2d_weight",
        )
        if bias:
            self.bias = Parameter((out_channel,), dtype=out_dtype, name="conv2d_bias")
        else:
            self.bias = None

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.groups == 1:
            if self.bias is not None:
                assert self.layout == "NCHW"
                outputs = conv2d_with_bias_nchw(
                    inputs.tensor,
                    self.weight.tensor,
                    self.bias.tensor,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.out_dtype,
                )

                conv_layer = layer(
                    outputs.op,
                    inputs=[inputs.tensor],
                    weights=[self.weight.tensor, self.bias.tensor],
                    const_scalars=None,
                    const_tensors=None,
                    requires_grad=self.training,
                    name=f"conv2d_{self.layout}_layer",
                )
            else:
                outputs = conv2d(
                    inputs.tensor,
                    self.weight.tensor,
                    self.stride,
                    self.padding,
                    self.dilation,
                    layout=self.layout,
                    out_dtype=self.out_dtype,
                )

                conv_layer = layer(
                    outputs.op,
                    inputs=[inputs.tensor],
                    weights=[self.weight.tensor],
                    const_scalars=None,
                    const_tensors=None,
                    requires_grad=self.training,
                    name=f"conv2d_{self.layout}_layer",
                )
        else:
            if self.bias is not None:
                assert self.layout == "NCHW"
                outputs = conv2d_with_group_nchw(
                    inputs.tensor,
                    self.weight.tensor,
                    self.bias.tensor,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    self.out_dtype,
                )
                conv_layer = layer(
                    outputs.op,
                    inputs=[inputs.tensor],
                    weights=[self.weight.tensor, self.bias.tensor],
                    const_scalars=None,
                    const_tensors=None,
                    requires_grad=self.training,
                    name=f"conv2d_{self.layout}_layer",
                )
            else:
                assert self.layout == "NCHW"
                outputs = conv2d_with_group_nchw(
                    inputs.tensor,
                    self.weight.tensor,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    self.out_dtype,
                )
                conv_layer = layer(
                    outputs.op,
                    inputs=[inputs.tensor],
                    weights=[self.weight.tensor],
                    const_scalars=None,
                    const_tensors=None,
                    requires_grad=self.training,
                    name=f"conv2d_{self.layout}_layer",
                )
        return conv_layer(inputs)


class CapsuleConv2d(Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        bias=False,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_caps=8,
        dtype="float32",
        out_dtype="float32",
        layout="NCHW",
    ):
        super(CapsuleConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
        stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, (list, tuple)) and len(stride) == 2
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, (tuple, list)) and len(padding) == 2
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
        assert isinstance(groups, int)
        assert isinstance(num_caps, int)
        assert layout in ["NCHW"]
        self.layout = layout

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_caps = num_caps

        assert groups == 1

        self.dtype = dtype
        self.out_dtype = out_dtype

        self.weight = Parameter(
            (out_channel, in_channel, *kernel_size, num_caps),
            dtype=dtype,
            name="conv2d_capsule_weight",
        )
        if bias:
            self.bias = Parameter(
                (out_channel, num_caps), dtype=out_dtype, name="conv2d_capsule_bias"
            )
        else:
            self.bias = None

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.bias is not None:
            assert self.layout == "NCHW"
            outputs = conv2d_capsule_nchw(
                inputs.tensor,
                self.weight.tensor,
                self.bias.tensor,
                self.stride,
                self.padding,
                self.dilation,
                self.num_caps,
                self.out_dtype,
            )
            conv_layer = layer(
                outputs.op,
                inputs=[inputs.tensor],
                weights=[self.weight.tensor, self.bias.tensor],
                const_scalars=None,
                const_tensors=None,
                requires_grad=self.training,
                name=f"conv2d_capsule_{self.layout}_layer",
            )
        else:
            assert self.layout == "NCHW"
            outputs = conv2d_capsule_nchw(
                inputs.tensor,
                self.weight.tensor,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.num_caps,
                self.out_dtype,
            )
            conv_layer = layer(
                outputs.op,
                inputs=[inputs.tensor],
                weights=[self.weight.tensor],
                const_scalars=None,
                const_tensors=None,
                requires_grad=self.training,
                name=f"conv2d_capsule_{self.layout}_layer",
            )
        return conv_layer(inputs)
