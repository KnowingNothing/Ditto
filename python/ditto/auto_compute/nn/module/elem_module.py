import tvm
from .module import Module, Parameter
from ..functional import add
from ...graph import LayerTensor, layer

ADD_LAYER = "add_layer"
ELEM_MODULE_REG = set([ADD_LAYER])


def is_elem_layer(layer):
    return layer.name in ELEM_MODULE_REG


class Add(Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, A, B):
        A, B = self.preprocess(A, B)
        outputs = add(A.tensor, B.tensor)

        add_layer = layer(
            outputs.op,
            inputs=[A.tensor, B.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name=ADD_LAYER,
        )
        return add_layer(A, B)
