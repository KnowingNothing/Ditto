import tvm
from ditto.auto_compute import layer, layer_tensor, graph
from ditto.autograd import grad_graph
from ditto.auto_compute.nn.module import Module, Parameter


class Pad(Module):
    def __init__(self):
        super(Pad, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        N, C, H, W = inputs.shape
        outputs = tvm.te.compute(
            [N, C, H + 4, W + 4],
            lambda n, c, h, w:
                tvm.tir.if_then_else(
                    tvm.tir.all(h > 2, h < H - 2, w > 2, w < W - 2),
                    inputs.tensor[n, c, h-2, w-2],
                    0.0
                ),
            name="pad"
        )

        pad_layer = layer(outputs.op, inputs=[inputs.tensor],
                           weights=None,
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name="pad_layer")
        return pad_layer(inputs)
    
    
class Dilate(Module):
    def __init__(self):
        super(Dilate, self).__init__()
        self.w = Parameter([64, 3, 3, 3], name="dilate_w")

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        N, C, H, W = inputs.shape
        rc = tvm.te.reduce_axis([0, C], "rc")
        rr = tvm.te.reduce_axis([0, 3], "rr")
        rs = tvm.te.reduce_axis([0, 3], "rs")
        outputs = tvm.te.compute(
            [N, 64, (H - 4) // 2, (W - 4) // 2],
            lambda n, k, p, q:
                tvm.te.sum(
                    inputs.tensor[n, rc, p*2 + rr*2, q*2 + rs*2] * self.w.tensor[k, rc, rr, rs],
                    axis=[rc, rr, rs]
                ),
            name="dilate"
        )

        dilate_layer = layer(outputs.op, inputs=[inputs.tensor],
                           weights=[self.w.tensor],
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name="dilate_layer")
        return dilate_layer(inputs)
    
    
    
class Depth2Space(Module):
    def __init__(self):
        super(Depth2Space, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        N, C, H, W = inputs.shape
        outputs = tvm.te.compute(
            [N, C//4, H * 4, W],
            lambda n, c, h, w:
                inputs.tensor[n, c*4+(h%4), h//4, w],
            name="depth2space_1"
        )
        
        outputs = tvm.te.compute(
            [N, C//4, H * 2, W * 2],
            lambda n, c, h, w:
                outputs[n, c, h*2+w%2, w//2],
            name="depth2space_2"
        )

        depth2space_layer = layer(outputs.op, inputs=[inputs.tensor],
                           weights=None,
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name="depth2space_layer")
        return depth2space_layer(inputs)
    
    
    
class Concat(Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x, y):
        x, y = self.preprocess(x, y)
        N, C1, H, W = x.shape
        _, C2, _, _ = y.shape
        outputs = tvm.te.compute(
            [N, C1 + C2, H, W],
            lambda n, c, h, w:
                tvm.tir.if_then_else(
                  c < C1,
                  x.tensor[n, c, h, w],
                  y.tensor[n, c-C1, h, w]  
                ),
            name="concat"
        )

        concat_layer = layer(outputs.op, inputs=[x.tensor, y.tensor],
                           weights=None,
                           const_scalars=None,
                           const_tensors=None,
                           requires_grad=self.training,
                           name="concat_layer")
        return concat_layer(x, y)
    
    

class Graph(Module):
    def __init__(self) -> None:
        super(Graph, self).__init__()
        
        self.pad = Pad()
        self.dilate = Dilate()
        self.d2s = Depth2Space()
        self.concat = Concat()
        
    def forward(self, inputs):
        x = self.pad(inputs)
        x = self.dilate(x)
        x = self.d2s(x)
        x = self.concat(inputs, x)
        return x
    

if __name__ == "__main__":
    model = Graph()
    A = layer_tensor([1, 3, 224, 224], name="A")
    outputs = model(A)
    g = graph([A], [outputs])
    grad_graph = grad_graph(g, reserve_forward=True)
    print(grad_graph)