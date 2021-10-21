import tvm._ffi
import tvm
from tvm.te import tensor
from . import _ffi_api
from tvm.runtime import Object


def layer_tensor(shape, name="layer_tensor", dtype="float32", layer=None, idx=0):
    t = tvm.te.placeholder(shape, name=name, dtype=dtype)
    return _ffi_api.LayerTensor(name, layer, t, idx)


def layer_tensor_from_te_tensor(t):
    return _ffi_api.LayerTensor(t.name, None, t, 0)


@tvm._ffi.register_object("ditto.auto_compute.LayerTensor")
class LayerTensor(Object):
    """LayerTensor object"""

    def __hash__(self):
        return _ffi_api.LayerTensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, LayerTensor):
            return False
        return _ffi_api.LayerTensorEqual(self, other)

    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def dtype(self):
        return self.tensor.dtype
    
    def __str__(self) -> str:
        ret = f"LayerTensor({self.name}, {self.shape}, {self.dtype})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.auto_compute.Layer")
class Layer(Object):
    """Layer object"""

    def __call__(self, *inputs):
        ninp = self.num_inputs
        if len(inputs) != ninp:
            raise ValueError(
                f"Need to provide {ninp} inputs, but get {len(inputs)}.")
        for inp in inputs:
            assert isinstance(inp, LayerTensor)
        ret = _ffi_api.ProduceOutputs(self, inputs)
        if len(ret) == 1:
            return ret[0]
    
    @property
    def all_ops(self):
        """Return all ops in this layer (including placeholder)"""
        return _ffi_api.LayerGetAllOps(self)
    
    @property
    def fingerprint(self):
        """Return the fingerprint of this layer"""
        return _ffi_api.LayerGetFingerprint(self)

    @property
    def num_inputs(self):
        """Number of inputs required by this layer."""
        return len(self.inputs)

    def __str__(self) -> str:
        all_ops = _ffi_api.LayerGetAllOps(self)
        ret = ""
        ret += f"====== Layer({self.name}) ======\n"
        ret += f"{len(all_ops)} ops in this layer.\n"
        ret += "inputs:\n"
        for inp in self.inputs:
            ret += str(inp)
            ret += "\n"
        ret += "weights:\n"
        for w in self.weights:
            ret += str(w)
            ret += "\n"
        ret += "---------------------------------\n"
        for op in all_ops:
            if hasattr(op, "body"):
                ret += op.name
                ret += ":  "
                ret += str(op.body[0])
                ret += "\n"
        ret += "---------------------------------\n"
        ret += "outputs:\n"
        for op in self.ops:
            ret += str(op.output(0))
            ret += "\n"
        ret += "=================================\n"
        return ret

    def __repr__(self) -> str:
        return str(self)


def layer(ops, inputs=None, weights=None, const_scalars=None,
          const_tensors=None, requires_grad=False, name="layer"):
    """Make a network layer through IR.

    Parameters
    ----------
    ops : te.Operation or List[te.Operation] (not empty)
        The output operations that make this layer.

    inputs : optional List[te.Tensor]
        The list of input tensors.

    weights : optional List[te.Tensor]
        The list of weights.

    const_scalars : optional List[PrimExpr]
        The list of constant scalar values.

    const_tensors : optional List[te.Tensor]
        The list of constant tensors.

    requires_grad : optional bool

    name: str

    Returns
    -------
    tensors: ditto.auto_compute.Layer
        The result layer

    Example
    -------
    .. code-block:: python

        x = tvm.te.placeholder((32, 3, 28, 28), name='x')
        w1 = tvm.te.placeholder((10, 3, 3, 3), name='w1')
        w2 = tvm.te.placeholder((10, 10, 3, 3), name='w2')
        z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
        z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
        y = topi.sum(z2)

        # make a layer
        layer = MakeLayer(y.op, inputs=[x], weights=[w1, w2])

    """
    if not isinstance(ops, list):
        ops = [ops]
    if inputs is None:
        inputs = []
    inputs = [x.tensor if isinstance(x, LayerTensor) else x for x in inputs]
    if weights is None:
        weights = []
    weights = [x.tensor if isinstance(x, LayerTensor) else x for x in weights]
    if const_scalars is None:
        const_scalars = []
    if const_tensors is None:
        const_tensors = []
    const_tensors = [x.tensor if isinstance(
        x, LayerTensor) else x for x in const_tensors]
    return _ffi_api.MakeLayer(name, ops, inputs, weights,
                              const_scalars, const_tensors)


# @tvm._ffi.register_object("ditto.auto_compute.Block")
# class Block(Object):
#     """Block object"""


# def block(out_tensors, name="block"):
#     if not isinstance(out_tensors, list):
#         out_tensors = [out_tensors]
#     return _ffi_api.Block(name, out_tensors)


@tvm._ffi.register_object("ditto.auto_compute.Graph")
class Graph(Object):
    """Graph object"""
    
    @property
    def all_layers(self):
        """Return all the layers in this graph"""
        return _ffi_api.GraphGetAllLayers(self)
    
    def __str__(self) -> str:
        all_layers = _ffi_api.GraphGetAllLayers(self)
        ret = ""
        ret += "*********************************\n"
        ret += f"****** Graph({self.name}) ******\n"
        ret += f"{len(all_layers)} layers in this graph.\n"
        ret += "inputs:\n"
        for inp in self.graph_inputs:
            ret += str(inp)
            ret += "\n"
        ret += "*********************************\n"
        for layer in all_layers:
            ret += str(layer)
        ret += "*********************************\n"
        ret += "outputs:\n"
        for out in self.graph_outputs:
            ret += str(out)
            ret += "\n"
        ret += "*********************************\n"
        ret += "*********************************\n"
        return ret

    def __repr__(self) -> str:
        return str(self)


def graph(graph_inputs, graph_outputs, name="graph"):
    """Make a network graph.

    Parameters
    ----------
    graph_inputs : optional List[ditto.auto_compute.LayerTensor]
        The list of input tensors.

    graph_outputs : optional List[ditto.auto_compute.LayerTensor]
        The list of output tensors.

    name: str

    Returns
    -------
    tensors: ditto.auto_compute.Graph
        The result graph
    """
    if not isinstance(graph_inputs, (list)):
        graph_inputs = [graph_inputs]
    if not isinstance(graph_outputs, (list)):
        graph_outputs = [graph_outputs]
    return _ffi_api.MakeGraph(name, graph_inputs, graph_outputs)
