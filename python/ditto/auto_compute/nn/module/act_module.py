import tvm
from .module import Module, Parameter
from ..functional import ReLU as relu, GELU as gelu
from ...graph import LayerTensor, layer


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = relu(inputs.tensor)

        relu_layer = layer(
            outputs.op,
            inputs=[inputs.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="relu_layer",
        )
        return relu_layer(inputs)


class GELU(Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = gelu(inputs.tensor)

        gelu_layer = layer(
            outputs.op,
            inputs=[inputs.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="gelu_layer",
        )
        return gelu_layer(inputs)
