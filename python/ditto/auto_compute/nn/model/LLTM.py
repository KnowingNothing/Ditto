import tvm
from ..module import Module, Parameter, Linear

from ...graph import layer


class ELU(Module):
    def __init__(self):
        super(ELU, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = tvm.te.compute(
            inputs.shape,
            lambda i, j: tvm.te.if_then_else(
                inputs[i, j] > 0, inputs[i, j], tvm.tir.exp(inputs[i, j]) - 1
            ),
            name="elu",
        )

        d_layer = layer(
            outputs.op,
            inputs=[inputs.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="elu_layer",
        )
        return d_layer(inputs)


class CatLayer(Module):
    def __init__(self):
        super(CatLayer, self).__init__()

    def forward(self, old_h, input):
        old_h, input = self.preprocess(old_h, input)
        batch_size, input_size = input.shape
        _, state_size = old_h.shape
        cat_size = input_size + state_size
        outputs = tvm.te.compute(
            [batch_size, cat_size],
            lambda i, j: tvm.te.if_then_else(
                j < input_size, input[i, j], old_h[i, j - input_size]
            ),
            name="cat",
        )

        d_layer = layer(
            outputs.op,
            inputs=[old_h.tensor, input.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="cat_layer",
        )
        return d_layer(old_h, input)


class SSELayer(Module):
    def __init__(self):
        super(SSELayer, self).__init__()

    def forward(self, gates):
        gates = self.preprocess(gates)
        batch_size, three_stsz = gates.shape
        two_stsz = three_stsz // 3 * 2
        outputs = tvm.te.compute(
            [batch_size, three_stsz],
            lambda i, j: tvm.te.if_then_else(
                j < two_stsz,
                tvm.te.sigmoid(gates[i, j]),
                tvm.te.if_then_else(
                    gates[i, j] > 0, gates[i, j], tvm.tir.exp(gates[i, j]) - 1
                ),
            ),
            name="sse",
        )

        d_layer = layer(
            outputs.op,
            inputs=[gates.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="sse_layer",
        )
        return d_layer(gates)


class NewCLayer(Module):
    def __init__(self):
        super(NewCLayer, self).__init__()

    def forward(self, gates_act, old_c):
        gates_act, old_c = self.preprocess(gates_act, old_c)
        batch_size, state_size = old_c.shape
        two_stsz = state_size * 2
        outputs = tvm.te.compute(
            [batch_size, state_size],
            lambda i, j: old_c[i, j] + gates_act[i, j + two_stsz] * gates_act[i, j],
            name="new_c_lltm",
        )

        d_layer = layer(
            outputs.op,
            inputs=[gates_act.tensor, old_c.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="new_c_layer",
        )
        return d_layer(gates_act, old_c)


class NewHLayer(Module):
    def __init__(self):
        super(NewHLayer, self).__init__()

    def forward(self, new_c, gates_act):
        new_c, gates_act = self.preprocess(new_c, gates_act)
        batch_size, state_size = new_c.shape
        outputs = tvm.te.compute(
            [batch_size, state_size],
            lambda i, j: tvm.te.tanh(new_c[i, j]) * gates_act[i, j + state_size],
            name="new_h_lltm",
        )

        d_layer = layer(
            outputs.op,
            inputs=[new_c.tensor, gates_act.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="new_h_layer",
        )
        return d_layer(new_c, gates_act)


class LLTM(Module):
    def __init__(self, state_size=128, input_size=28 * 28, dtype="float32"):
        super(LLTM, self).__init__()
        self.l1 = Linear(
            state_size + input_size,
            3 * state_size,
            bias=True,
            dtype=dtype,
            out_dtype=dtype,
        )
        self.out = Linear(state_size, 10, bias=True, dtype=dtype, out_dtype=dtype)
        self.cat = CatLayer()
        self.sse = SSELayer()
        self.new_c = NewCLayer()
        self.new_h = NewHLayer()

    def forward(self, x, old_h, old_c):
        X = self.cat(old_h, x)
        gates = self.l1(X)
        gates_act = self.sse(gates)
        new_c = self.new_c(gates_act, old_c)
        new_h = self.new_h(new_c, gates_act)
        result = self.out(new_h)
        return result, new_h, new_c
