import tvm
from ..module import Module, Parameter, Linear

from ...graph import layer


class InternalGateLayer(Module):
    def __init__(self, hidden_size, activation, dtype="float32"):
        super(InternalGateLayer, self).__init__()
        if activation == "sigmoid":
            self.act = tvm.te.sigmoid
        elif activation == "tanh":
            self.act = tvm.te.tanh
        else:
            raise ValueError(f"Unsupported activation {activation}.\n")

        self.alpha = Parameter([hidden_size], dtype=dtype, name="alpha_f")
        self.beta1 = Parameter([hidden_size], dtype=dtype, name="beta_f1")
        self.beta2 = Parameter([hidden_size], dtype=dtype, name="beta_f2")

    def forward(self, xi, hh):
        xi, hh = self.preprocess(xi, hh)
        batch, hidden_size = xi.shape
        b_, h_ = hh.shape
        assert int(b_) == int(batch) and int(h_) == int(hidden_size)
        outputs = tvm.te.compute(
            [batch, hidden_size],
            lambda i, j: self.act(
                self.alpha[j] * xi[i, j] * hh[i, j]
                + (self.beta1[j] * xi[i, j])
                + (self.beta2[j] * hh[i, j])
            ),
            name="gate",
        )

        d_layer = layer(
            outputs.op,
            inputs=[xi.tensor, hh.tensor],
            weights=[self.alpha.tensor, self.beta1.tensor, self.beta2.tensor],
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="gate_layer",
        )
        return d_layer(xi, hh)


class CXLayer(Module):
    def __init__(self):
        super(CXLayer, self).__init__()

    def forward(self, f_g, old_c, i_g, z_t):
        f_g, old_c, i_g, z_t = self.preprocess(f_g, old_c, i_g, z_t)
        batch, hidden_size = old_c.shape
        outputs = tvm.te.compute(
            [batch, hidden_size],
            lambda i, j: f_g[i, j] * old_c[i, j] + i_g[i, j] * z_t[i, j],
            name="compute_cx",
        )

        d_layer = layer(
            outputs.op,
            inputs=[f_g.tensor, old_c.tensor, i_g.tensor, z_t.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="cx_layer",
        )
        return d_layer(f_g, old_c, i_g, z_t)


class HXLayer(Module):
    def __init__(self):
        super(HXLayer, self).__init__()

    def forward(self, o_g, cx):
        o_g, cx = self.preprocess(o_g, cx)
        batch, hidden_size = o_g.shape
        b_, h_ = cx.shape
        assert b_ == batch and hidden_size == h_
        outputs = tvm.te.compute(
            [batch, hidden_size],
            lambda i, j: o_g[i, j] * tvm.te.tanh(cx[i, j]),
            name="compute_hx",
        )

        d_layer = layer(
            outputs.op,
            inputs=[o_g.tensor, cx.tensor],
            weights=None,
            const_scalars=None,
            const_tensors=None,
            requires_grad=self.training,
            name="hx_layer",
        )
        return d_layer(o_g, cx)


class MI_LSTM(Module):
    def __init__(
        self,
        input_size=28 * 28,
        hidden_size=1024,
        n_class=10,
        dtype="float32",
        out_dtype="float32",
    ):
        super(MI_LSTM, self).__init__()
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.weight_fh = Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_ih = Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_zh = Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_oh = Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_fx = Linear(
            input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_ix = Linear(
            input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_zx = Linear(
            input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )
        self.weight_ox = Linear(
            input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype
        )

        self.gate1 = InternalGateLayer(hidden_size, "sigmoid", dtype=self.out_dtype)
        self.gate2 = InternalGateLayer(hidden_size, "sigmoid", dtype=self.out_dtype)
        self.gate3 = InternalGateLayer(hidden_size, "sigmoid", dtype=self.out_dtype)
        self.gate4 = InternalGateLayer(hidden_size, "tanh", dtype=self.out_dtype)

        self.cx = CXLayer()
        self.hx = HXLayer()

        self.out = Linear(
            hidden_size, 10, dtype=self.out_dtype, out_dtype=self.out_dtype
        )

    def forward(self, inp, old_h, old_c):
        fxi = self.weight_fx(inp)
        fhh = self.weight_fh(old_h)
        f_g = self.gate1(fxi, fhh)

        ixi = self.weight_ix(inp)
        ihh = self.weight_ih(old_h)
        i_g = self.gate2(ixi, ihh)

        oxi = self.weight_ox(inp)
        ohh = self.weight_oh(old_h)
        o_g = self.gate3(oxi, ohh)

        zxi = self.weight_zx(inp)
        zhh = self.weight_zh(old_h)
        z_t = self.gate4(zxi, zhh)

        cx = self.cx(f_g, old_c, i_g, z_t)
        hx = self.hx(o_g, cx)

        result = self.out(hx)
        return result, hx, cx
