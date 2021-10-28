import tvm
from ..functional import softmax
from ..module import (
    Module,
    Parameter,
    Conv2d,
    CapsuleConv2d,
    BatchNorm2d,
    ReLU,
    AvgPool2d,
    GlobalAvgPool2d,
    Linear,
    Sequential,
    Add
)
from ...graph import layer


class ConvLayer(Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, dtype="float32", out_dtype="float32"):
        super(ConvLayer, self).__init__()

        self.conv = Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dtype=dtype,
                              out_dtype=out_dtype
                              )
        self.relu = ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))
    
    
class SquashLayer(Module):
    def __init__(self):
        super(SquashLayer, self).__init__()
    
    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        N, C, H, W, Cap = inputs.shape
        
        rc = tvm.te.reduce_axis([0, C], "rc")
        rh = tvm.te.reduce_axis([0, H], "rh")
        rw = tvm.te.reduce_axis([0, W], "rw")
        rcap = tvm.te.reduce_axis([0, Cap], "rcap")
        squared_norm = tvm.te.compute(
            [N],
            lambda n:
                tvm.te.sum(
                    inputs[n, rc, rh, rw, rcap] * inputs[n, rc, rh, rw, rcap],
                    axis=[rc, rh, rw, rcap]
                ),
            name="squared_norm"
        )
        
        output_tensor = tvm.te.compute(
            [N, C, H, W, Cap],
            lambda n, c, h, w, p:
                squared_norm[n] * inputs[n, c, h, w, p] / (1. + squared_norm[n]) * tvm.te.sqrt(squared_norm[n]),
            name="output_tensor"
        )
        
        sq_layer = layer(output_tensor.op, inputs=[inputs.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="squash_layer")
        return sq_layer(inputs)


class PrimaryCaps(Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6, dtype="float32", out_dtype="float32"):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = CapsuleConv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=False,
            stride=2,
            padding=0,
            num_caps=num_capsules,
            dtype=dtype,
            out_dtype=out_dtype
        )
        self.squash = SquashLayer()
        
    def forward(self, x):
        u = self.capsules(x)
        return self.squash(u)


class DigitCaps(Module):
    def __init__(self, num_capsules=10, num_channel=32, height=6, width=6, in_channels=8, out_channels=16, dtype="float32", out_dtype="float32"):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules

        self.W = Parameter((num_channel, height, width, num_capsules, out_channels, in_channels),
                           dtype=dtype, name="digit_cap_weight")
        
    def forward(self, x):
        x = self.preprocess(x)
        N, C, H, W, Cap = x.shape
        
        rr = tvm.te.reduce_axis([0, Cap], "rr")
        outputs = tvm.te.compute(
            [N, C, H, W, self.num_capsules, self.out_channels],
            lambda n, c, h, w, cap, o:
                tvm.te.sum(
                    x[n, c, h, w, rr] * self.W[c, h, w, cap, o, rr],
                    axis=[rr]
                ),
            name="digit_capsule"
        )
        d_layer = layer(outputs.op, inputs=[x.tensor],
                               weights=[self.W.tensor],
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="digit_cap_layer")
        return d_layer(x)


class ExpandWeightLayer(Module):
    def __init__(self, channel, height, width, cap_h, dtype="float32"):
        super(ExpandWeightLayer, self).__init__()
        self.b_ij = Parameter(
            [channel, height, width, cap_h],
            name="b_ij",
            dtype=dtype
        )
    
    def forward(self, batch):
        x = self.b_ij.tensor
        C, H, W, Cap = x.shape
        outputs = tvm.te.compute(
            [batch, C, H, W, Cap],
            lambda n, c, h, w, p:
                x[c, h, w, p],
            "expand"
        )
        
        dy_layer = layer(outputs.op, inputs=[],
                               weights=[x],
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="expand_weight_layer")
        return dy_layer()

class SoftMaxLayer(Module):
    def __init__(self):
        super(SoftMaxLayer, self).__init__()
    
    def forward(self, x):
        x = self.preprocess(x)
        N, C, H, W, Cap = x.shape
        rc = tvm.te.reduce_axis([0, C], "rc")
        rh = tvm.te.reduce_axis([0, H], "rh")
        rw = tvm.te.reduce_axis([0, W], "rw")
        tmp = tvm.te.compute(
            [N, Cap],
            lambda n, p:
                tvm.te.sum(
                    tvm.te.exp(x[n, rc, rh, rw, p]),
                    axis=[rc, rh, rw]
                ),
            "sum_exp"
        )
        outputs = tvm.te.compute(
            [N, C, H, W, Cap],
            lambda n, c, h, w, p:
                tvm.te.exp(x[n, c, h, w, p]) / tmp[n, p],
            "softmax"
        )
        
        dy_layer = layer(outputs.op, inputs=[x.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="softmax_layer")
        return dy_layer(x)
    
    
class MulCijLayer(Module):
    def __init__(self):
        super(MulCijLayer, self).__init__()
        
    def forward(self, x, y):
        x, y = self.preprocess(x, y)
        N, C, H, W, Cap1, Cap2 = y.shape
        rc = tvm.te.reduce_axis([0, C], "rc")
        rh = tvm.te.reduce_axis([0, H], "rh")
        rw = tvm.te.reduce_axis([0, W], "rw")
        outputs = tvm.te.compute(
            [N, Cap1, Cap2],
            lambda n, p1, p2:
                tvm.te.sum(
                     x[n, rc, rh, rw, p1] * y[n, rc, rh, rw, p1, p2],
                     axis=[rc, rh, rw]
                ),               
            "mul_cij_u_hat"
        )
        dy_layer = layer(outputs.op, inputs=[y.tensor, x.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="mul_cij_layer")
        return dy_layer(y, x)
    
    
class MulVjLayer(Module):
    def __init__(self):
        super(MulVjLayer, self).__init__()
        
    def forward(self, x, y):
        x, y = self.preprocess(x, y)
        N, C, H, W, Cap1, Cap2 = x.shape
        N_, Cap1_, Cap2_ = y.shape
        rp = tvm.te.reduce_axis([0, Cap2], "rp")
        outputs = tvm.te.compute(
            [N, C, H, W, Cap1],
            lambda n, c, h, w, p:
                tvm.te.sum(
                    x[n, c, h, w, p, rp] * y[n, p, rp],
                    axis=[rp]
                ),
            "mul_vj"
        )
        dy_layer = layer(outputs.op, inputs=[x.tensor, y.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="mul_vj_layer")
        return dy_layer(x, y)
    
    
class AddLayer(Module):
    def __init__(self):
        super(AddLayer, self).__init__()
    
    def forward(self, x, y):
        x, y = self.preprocess(x, y)
        N, C, H, W, Cap1 = y.shape
        N_, C_, H_, W_, Cap1_ = x.shape
        outputs = tvm.te.compute(
            [N, C, H, W, Cap1],
            lambda n, c, h, w, p1:
                x[n, c, h, w, p1] + y[n, c, h, w, p1],
            "add_bij"
        )
        dy_layer = layer(outputs.op, inputs=[x.tensor, y.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="add_layer")
        return dy_layer(x, y)
    

class ActSquashLayer(Module):
    def __init__(self):
        super(ActSquashLayer, self).__init__()
        
    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        N, Cap1, Cap2 = inputs.shape
        
        outputs = tvm.te.compute(
            [N, Cap1, Cap2],
            lambda n, p1, p2:
                inputs[n, p1, p2] * inputs[n, p1, p2] * inputs[n, p1, p2] / (1. + inputs[n, p1, p2] * inputs[n, p1, p2]) * inputs[n, p1, p2],
            name="output_tensor"
        )
        
        dy_layer = layer(outputs.op, inputs=[inputs.tensor],
                               weights=None,
                               const_scalars=None,
                               const_tensors=None,
                               requires_grad=self.training,
                               name="act_squash_layer")
        return dy_layer(inputs)


class DynamicRouting(Module):
    def __init__(self, channel=32, height=6, width=6, cap_h=10, cap_w=16, dtype="float32", out_dtype="float32"):
        super(DynamicRouting, self).__init__()
        self.expand = ExpandWeightLayer(channel, height, width, cap_h, dtype)
        self.softmax = SoftMaxLayer()
        self.mul_cij = MulCijLayer()
        self.squash = ActSquashLayer()
        self.mul_vj = MulVjLayer()
        self.add = AddLayer()
        self.dtype = dtype
        self.out_dtype = out_dtype
        
    def forward(self, inputs):
        num_iterations = 3
        b_ij = self.expand(inputs.shape[0])
        for iteration in range(num_iterations):
            c_ij = self.softmax(b_ij)
            s_j = self.mul_cij(c_ij, inputs)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = self.mul_vj(inputs, v_j)
                b_ij = self.add(b_ij, a_ij)
                print(b_ij.shape)

        return v_j


class CapsNet(Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.dynamic_routing = DynamicRouting()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        output = self.dynamic_routing(output)
        return output
