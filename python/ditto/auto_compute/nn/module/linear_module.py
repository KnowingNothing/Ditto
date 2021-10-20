import tvm
from .module import Module, Parameter
from ..functional import linear
from ...graph import LayerTensor, layer, layer_tensor


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False,
                 dtype="float32", out_dtype="float32"):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            [out_features, in_features], dtype=dtype, name="linear_weight")
        if bias:
            self.bias = Parameter(
                [out_features], dtype=out_dtype, name="linear_bias")
        else:
            self.bias = None

        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.bias is not None:
            outputs = linear(
                inputs.tensor,
                self.weight.tensor,
                self.bias.tensor,
                out_dtype=self.out_dtype)
            linear_layer = layer(outputs.op, inputs=[inputs.tensor],
                                 weights=[self.weight.tensor,
                                          self.bias.tensor],
                                 requires_grad=self.training,
                                 name="linear_layer")
        else:
            outputs = linear(
                inputs.tensor,
                self.weight.tensor,
                out_dtype=self.out_dtype)
            linear_layer = layer(outputs.op, inputs=[inputs.tensor],
                                 weights=[self.weight.tensor],
                                 requires_grad=self.training,
                                 name="linear_layer")

        return linear_layer(inputs)
