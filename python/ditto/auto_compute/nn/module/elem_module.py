import tvm
from .module import Module, Parameter
from ..functional import add
from ...graph import LayerTensor, layer


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
            name="add_layer",
        )
        return add_layer(A, B)
