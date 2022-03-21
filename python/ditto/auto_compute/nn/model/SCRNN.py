import tvm
from ..module import Module, Parameter, Linear

from ...graph import layer


class InternalLayer1(Module):
    def __init__(self, alpha=0.5):
        super(InternalLayer1, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, state_c):
        inputs, state_c = self.preprocess(inputs, state_c)
        batch, input_size = inputs.shape
        __, context_units = state_c.shape
        context_state = tvm.te.compute(
            [batch, context_units],
            lambda i, j: (1 - self.alpha) * inputs[i, j] + self.alpha * state_c[i, j],
            name="state_c",
        )

        d_layer = layer(
            context_state.op,
            inputs=[inputs.tensor, state_c.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="interanal1_layer",
        )
        return d_layer(inputs, state_c)


class ConcatLayer(Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, context_state, inputs, state_h):
        context_state, inpust, state_h = self.preprocess(context_state, inputs, state_h)
        batch, context_units = context_state.shape
        batch, input_size = inputs.shape
        _, num_units = state_h.shape
        concated = tvm.te.compute(
            [batch, context_units + input_size + num_units],
            lambda i, j: tvm.te.if_then_else(
                j < context_units,
                context_state[i, j],
                tvm.te.if_then_else(
                    j < context_units + input_size,
                    inputs[i, j - context_units],
                    state_h[i, j - context_units - input_size],
                ),
            ),
            name="concated",
        )

        d_layer = layer(
            concated.op,
            inputs=[context_state.tensor, inputs.tensor, state_h.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="interanal1_layer",
        )
        return d_layer(context_state, inputs, state_h)


class InternalLayer2(Module):
    def __init__(self):
        super(InternalLayer2, self).__init__()

    def forward(self, fc):
        fc = self.preprocess(fc)
        N, H = fc.shape
        outputs = tvm.te.compute(
            [N, H], lambda i, j: tvm.te.sigmoid(fc[i, j]), name="sigmoid"
        )

        d_layer = layer(
            outputs.op,
            inputs=[fc.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="interanal2_layer",
        )
        return d_layer(fc)


class InternalLayer3(Module):
    def __init__(self):
        super(InternalLayer3, self).__init__()

    def forward(self, h, c):
        h, c = self.preprocess(h, c)
        batch, num_units = h.shape
        outputs = tvm.te.compute(
            [batch, num_units], lambda i, j: h[i, j] + c[i, j], name="new_h"
        )

        d_layer = layer(
            outputs.op,
            inputs=[h.tensor, c.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="interanal3_layer",
        )
        return d_layer(h, c)


class SCRNN(Module):
    def __init__(
        self,
        num_units=128,
        context_units=64,
        input_size=28 * 28,
        dtype="float32",
        out_dtype="float32",
    ):
        super(SCRNN, self).__init__()
        # B : [28*28, 64]
        # U: [128, 128]
        # V: [64, 128]
        # FC layer 64+28*28+128 -> 128
        self.B = Linear(input_size, context_units, dtype=dtype, out_dtype=out_dtype)
        self.U = Linear(num_units, num_units, dtype=out_dtype, out_dtype=out_dtype)
        self.V = Linear(context_units, num_units, dtype=out_dtype, out_dtype=out_dtype)
        self.fc = Linear(
            num_units + context_units + input_size,
            num_units,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.out = Linear(num_units, 10, dtype=out_dtype, out_dtype=out_dtype)

        self.inter1 = InternalLayer1()
        self.inter2 = InternalLayer2()
        self.inter3 = InternalLayer3()

        self.cat = ConcatLayer()

    def forward(self, x, old_h, old_c):
        # state_h : [batch, 128=num_units]
        # state_c : [batch, 64=context_units]
        input_b = self.B(x)
        context_state = self.inter1(input_b, old_c)
        concated = self.cat(context_state, x, old_h)
        fc_ed = self.fc(concated)
        hidden_state = self.inter2(fc_ed)
        hU = self.U(hidden_state)
        cV = self.V(context_state)
        new_h = self.inter3(hU, cV)
        result = self.out(new_h)
        return result, new_h, context_state
