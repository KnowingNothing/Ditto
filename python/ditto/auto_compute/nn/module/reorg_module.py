import tvm
from .module import Module, Parameter
from ..functional import shuffle_channels, batch_flatten, cat_channel
from ...graph import LayerTensor, layer


class ShuffleChannel(Module):
    def __init__(self, groups):
        super(ShuffleChannel, self).__init__()
        self.groups = groups

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = shuffle_channels(inputs.tensor, self.groups)

        shuffle_layer = layer(outputs.op, inputs=[inputs.tensor],
                              weights=None,
                              const_scalars=None,
                              const_tensors=None,
                              requires_grad=self.training,
                              name="shuffle_channel_layer")
        return shuffle_layer(inputs)


class BatchFlatten(Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = batch_flatten(inputs.tensor)

        flatten_layer = layer(outputs.op, inputs=[inputs.tensor],
                              weights=None,
                              const_scalars=None,
                              const_tensors=None,
                              requires_grad=self.training,
                              name="batch_flatten_layer")
        return flatten_layer(inputs)


class CatChannel(Module):
    def __init__(self):
        super(CatChannel, self).__init__()

    def forward(self, A, B):
        A, B = self.preprocess(A, B)
        outputs = cat_channel(A.tensor, B.tensor)

        cat_layer = layer(outputs.op, inputs=[A.tensor, B.tensor],
                          weights=None,
                          const_scalars=None,
                          const_tensors=None,
                          requires_grad=self.training,
                          name="cat_channel_layer")
        return cat_layer(A, B)
