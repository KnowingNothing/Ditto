import tvm
from ..module import (
    Module,
    Parameter,
    Linear
)

from ...graph import layer


class GateLayer(Module):
    def __init__(self):
        super(GateLayer, self).__init__()

    def forward(self, input_m_weight, oldh_m_weight):
        input_m_weight, oldh_m_weight = self.preprocess(
            input_m_weight, oldh_m_weight)
        batch_size, four_stsz = input_m_weight.shape
        outputs = tvm.te.compute(
            [batch_size, four_stsz],
            lambda i, j: tvm.te.sigmoid
            (
                input_m_weight[i, j]
                + oldh_m_weight[i, j]
            ),
            name="gate_sublstm"
        )

        d_layer = layer(outputs.op, inputs=[input_m_weight.tensor, oldh_m_weight.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="gate_layer")
        return d_layer(input_m_weight, oldh_m_weight)


class NewCLayer(Module):
    def __init__(self):
        super(NewCLayer, self).__init__()

    def forward(self, gates, old_c):
        gates, old_c = self.preprocess(gates, old_c)
        batch_size, state_size = old_c.shape
        outputs = tvm.te.compute(
            [batch_size, state_size],
            lambda i, j: gates[i, state_size + j] * old_c[i,
                                                          j] + gates[i, j + 2*state_size] - gates[i, j],
            name="new_c_sublstm"
        )

        d_layer = layer(outputs.op, inputs=[gates.tensor, old_c.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="new_c_layer")
        return d_layer(gates, old_c)


class NewHLayer(Module):
    def __init__(self):
        super(NewHLayer, self).__init__()

    def forward(self, new_c, gates):
        new_c, gates = self.preprocess(new_c, gates)
        batch_size, state_size = new_c.shape
        outputs = tvm.te.compute(
            [batch_size, state_size],
            lambda i, j: tvm.te.sigmoid(
                new_c[i, j]) - gates[i, j + 3*state_size],
            name="new_h_sublstm"
        )

        d_layer = layer(outputs.op, inputs=[new_c.tensor, gates.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="new_h_layer")
        return d_layer(new_c, gates)


class subLSTM(Module):
    def __init__(self, state_size=128, input_size=28*28, dtype="float32", out_dtype="float32"):
        super(subLSTM, self).__init__()
        self.l1 = Linear(input_size, 4*state_size, bias=True,
                         dtype=dtype, out_dtype=out_dtype)
        self.l2 = Linear(state_size, 4*state_size, bias=True,
                         dtype=dtype, out_dtype=out_dtype)
        self.g1 = GateLayer()
        self.new_c = NewCLayer()
        self.new_h = NewHLayer()
        self.out = Linear(state_size, 10, bias=True,
                          dtype=out_dtype, out_dtype=out_dtype)

    def forward(self, x, old_h, old_c):
        input_m_weight = self.l1(x)
        oldh_m_weight = self.l2(old_h)
        gates = self.g1(input_m_weight, oldh_m_weight)
        new_c = self.new_c(gates, old_c)
        new_h = self.new_h(new_c, gates)
        result = self.out(new_h)
        return result, new_h, new_c
