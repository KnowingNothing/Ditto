import tvm._ffi
import tvm
from tvm.te import tensor
from . import _ffi_api
from tvm.runtime import Object


def layer_tensor(shape, name="layer_tensor", dtype="float32"):
    t = tvm.te.placeholder(shape, name=name, dtype=dtype)
    return _ffi_api.LayerTensor(name, None, t, 0)


@tvm._ffi.register_object("ditto.LayerTensor")
class LayerTensor(Object):
    """LayerTensor object"""

    def __hash__(self):
        return _ffi_api.LayerTensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, LayerTensor):
            return False
        return _ffi_api.LayerTensorEqual(self, other)


def create_op_state(op):
    """Make op state"""
    return _ffi_api.CreateOpState(op)


@tvm._ffi.register_object("ditto.Layer")
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
    def num_inputs(self):
        """Number of inputs required by this layer."""
        return len(self.inputs)


def layer(ops, inputs=None, weights=None, const_scalars=None,
          const_tensors=None, gradients=None, requires_grad=False, name="layer"):
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

    gradients : optional List[te.Tensor]
        The list of gradients.

    requires_grad : optional bool
        If gradients is None and requires_grad is set True, autodiff is used
        to calculate the gradients for this layer.

    name: str

    Returns
    -------
    tensors: nast.Layer
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
    if weights is None:
        weights = []
    if const_scalars is None:
        const_scalars = []
    if const_tensors is None:
        const_tensors = []
    if gradients is None:
        if requires_grad:
            # TODO: integrate autodiff into this function
            raise RuntimeError("Currently not support autodiff in MakeLayer")
        else:
            gradients = []
    return _ffi_api.MakeLayer(name, ops, inputs, weights,
                              const_scalars, const_tensors, gradients)


def create_layer_state(layer):
    """Make the Layer State"""
    return _ffi_api.CreateLayerState(layer)


@tvm._ffi.register_object("ditto.Block")
class Block(Object):
    """Block object"""


def block(out_tensors, name="block"):
    if not isinstance(out_tensors, list):
        out_tensors = [out_tensors]
    return _ffi_api.Block(name, out_tensors)


@tvm._ffi.register_object("ditto.Graph")
class Graph(Object):
    """Graph object"""


def graph(blocks, name="graph"):
    if not isinstance(blocks, list):
        blocks = [blocks]
    return _ffi_api.Graph(name, blocks)


@tvm._ffi.register_object("ditto.TensorState")
class TensorState(Object):
    """TensorState object"""


@tvm._ffi.register_object("ditto.OpState")
class OpState(Object):
    """OpState object"""

    def axis(self):
        return _ffi_api.OpStateGetAxis(self)

    def reduce_axis(self):
        return _ffi_api.OpStateGetReduceAxis(self)

    def transform(self, spatial_forward, fspatial_backward,
                  reduce_forward, freduce_backward):
        def _make_vars(forward, fcompute):
            code = fcompute.__code__
            names = code.co_varnames
            if not (len(forward) == len(names)):
                raise ValueError(f"transform var number mismatch {len(forward)} vs {len(names)}.\n")
            vvars = [tvm.tir.Var(name, "int32") for name in names]
            return vvars, fcompute(*vvars)

        spatial_vars, spatial_backward = _make_vars(
            spatial_forward, fspatial_backward)
        reduce_vars, reduce_backward = _make_vars(
            reduce_forward, freduce_backward)
        return _ffi_api.OpStateTransform(self, spatial_vars, spatial_forward,
                                         spatial_backward, reduce_vars,
                                         reduce_forward, reduce_backward)


@tvm._ffi.register_object("ditto.LayerState")
class LayerState(Object):
    """LayerState object"""

    def __getitem__(self, k):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")
        return _ffi_api.LayerStateGetOpState(self, k)

    def make_compute(self, layer_inputs):
        return _ffi_api.LayerStateMakeCompute(self, layer_inputs)
