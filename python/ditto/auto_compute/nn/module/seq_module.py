import tvm
from .module import Module, Parameter
from ...graph import LayerTensor, layer


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for i, arg in enumerate(args):
            setattr(self, "seq_" + str(i), arg)
        self.num_args = len(args)

    def forward(self, x):
        for i in range(self.num_args):
            x = getattr(self, "seq_" + str(i))(x)
        return x
