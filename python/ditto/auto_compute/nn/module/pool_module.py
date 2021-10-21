import tvm
from .module import Module, Parameter
from ..functional import avgpool2d_nchw, global_avgpool2d_nchw
from ...graph import LayerTensor, layer


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, layout="NCHW"):
        super(AvgPool2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        assert isinstance(self.kernel_size, (tuple, list)
                          ) and len(self.kernel_size) == 2
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(self.stride, (list, tuple)) and len(self.stride) == 2
        self.padding = padding
        self.layout = layout

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.layout == "NCHW":
            outputs = avgpool2d_nchw(
                inputs.tensor,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                stride_h=self.stride[0],
                stride_w=self.stride[1],
                padding=self.padding)
        else:
            raise NotImplementedError(
                f"No layout {self.layout} for avgpool.\n")

        pool_layer = layer(outputs.op, inputs=[inputs.tensor],
                           weights=None,
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name=f"avgpool2d_{self.layout}_layer")
        ret = pool_layer(inputs)
        return ret


class GlobalAvgPool2d(Module):
    def __init__(self, keep_dim=False, layout="NCHW"):
        super(GlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim
        self.layout = layout

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.layout == "NCHW":
            outputs = global_avgpool2d_nchw(
                inputs.tensor,
                self.keep_dim)
        else:
            raise NotImplementedError(
                f"No layout {self.layout} for avgpool.\n")

        pool_layer = layer(outputs.op, inputs=[inputs.tensor],
                           weights=None,
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name=f"global_avgpool2d_{self.layout}_layer")
        return pool_layer(inputs)
