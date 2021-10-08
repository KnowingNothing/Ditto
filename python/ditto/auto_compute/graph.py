import tvm._ffi
import tvm
from tvm.te import tensor
from . import _ffi_api
from tvm.runtime import Object


def layer_tensor(shape, name="layer_tensor", dtype="float32"):
    t = tvm.te.placeholder(shape, name=name, dtype=dtype)
    return _ffi_api.LayerTensor(name, None, t, 0)


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


def create_op_state(op):
    """Make op state"""
    return _ffi_api.CreateOpState(op)


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


@tvm._ffi.register_object("ditto.auto_compute.Block")
class Block(Object):
    """Block object"""


def block(out_tensors, name="block"):
    if not isinstance(out_tensors, list):
        out_tensors = [out_tensors]
    return _ffi_api.Block(name, out_tensors)


@tvm._ffi.register_object("ditto.auto_compute.Graph")
class Graph(Object):
    """Graph object"""


def graph(blocks, name="graph"):
    if not isinstance(blocks, list):
        blocks = [blocks]
    return _ffi_api.Graph(name, blocks)


@tvm._ffi.register_object("ditto.auto_compute.TensorState")
class TensorState(Object):
    """TensorState object"""


@tvm._ffi.register_object("ditto.auto_compute.OpState")
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
                raise ValueError(
                    f"transform var number mismatch {len(forward)} vs {len(names)}.\n")
            vvars = [tvm.tir.Var(name, "int32") for name in names]
            return vvars, fcompute(*vvars)

        spatial_vars, spatial_backward = _make_vars(
            spatial_forward, fspatial_backward)
        reduce_vars, reduce_backward = _make_vars(
            reduce_forward, freduce_backward)
        return _ffi_api.OpStateTransform(self, spatial_vars, spatial_forward,
                                         spatial_backward, reduce_vars,
                                         reduce_forward, reduce_backward)


@tvm._ffi.register_object("ditto.auto_compute.LayerState")
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

    def transform(self, k, spatial_forward, fspatial_backward,
                  reduce_forward, freduce_backward, explicit_transform=True):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")

        def _make_vars(forward, fcompute):
            code = fcompute.__code__
            names = code.co_varnames
            if not (len(forward) == len(names)):
                raise ValueError(
                    f"transform var number mismatch {len(forward)} vs {len(names)}.\n")
            vvars = [tvm.tir.Var(name, "int32") for name in names]
            return vvars, fcompute(*vvars)

        spatial_vars, spatial_backward = _make_vars(
            spatial_forward, fspatial_backward)
        reduce_vars, reduce_backward = _make_vars(
            reduce_forward, freduce_backward)
        ret = _ffi_api.LayerStateTransform(self, k, spatial_vars, spatial_forward,
                                            spatial_backward, reduce_vars,
                                            reduce_forward, reduce_backward,
                                            1 if explicit_transform else 0)
        return ret.output(0)

    def _fold(self, k, axis, factor, explicit=True):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")
        if not isinstance(axis, tvm.tir.IterVar):
            raise ValueError("Expect axis to be IterVar")
        if not isinstance(factor, (int, tvm.tir.IntImm)):
            raise ValueError("Expect axis to be int or IntImm")

        def _inner(vars, forward, backward, ivs):
            for iv in ivs:
                if iv == axis:
                    outer = tvm.tir.Var(axis.var.name + ".outer", "int32")
                    inner = tvm.tir.Var(axis.var.name + ".inner", "int32")
                    vars.extend([outer, inner])
                    forward.extend([iv.var//factor, iv.var % factor])
                    backward.append(outer*factor + inner)
                else:
                    var = tvm.tir.Var(iv.var.name, "int32")
                    vars.append(var)
                    forward.append(iv.var)
                    backward.append(var)

        spatial_vars = []
        spatial_forward = []
        spatial_backward = []
        _inner(spatial_vars, spatial_forward, spatial_backward, self[k].axis())

        reduce_vars = []
        reduce_forward = []
        reduce_backward = []
        _inner(reduce_vars, reduce_forward,
               reduce_backward, self[k].reduce_axis())

        ret = _ffi_api.LayerStateTransform(self, k, spatial_vars, spatial_forward,
                                            spatial_backward, reduce_vars,
                                            reduce_forward, reduce_backward,
                                            1 if explicit else 0)
        return ret.output(0)
    
    def _shuffle(self, k, *axes, explicit=True):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")
        visit = set()
        for i, iv in enumerate(axes):
            assert not iv in visit, f"Repeated iter_var in shuffle: {iv}.\n"
            assert iv.iter_type == 0, f"Only expect spatial axis in shuffle.\n"
            visit.add(iv)
        select_mapping = {}
        for iv in self[k].axis():
            var = tvm.tir.Var(iv.var.name, "int32")
            select_mapping[iv] = var
        for iv in axes:
            assert iv in select_mapping, f"axis {iv} is not part of the op {k}.\n"

        pos = 0
        spatial_vars = []
        spatial_forward = []
        spatial_backward = []
        for i, iv in enumerate(self[k].axis()):
            if iv in visit:
                spatial_vars.append(select_mapping[axes[pos]])
                spatial_forward.append(axes[pos].var)
                spatial_backward.append(select_mapping[iv])
                pos += 1
            else:
                spatial_vars.append(select_mapping[iv])
                spatial_forward.append(iv.var)
                spatial_backward.append(select_mapping[iv])
        assert pos == len(visit), f"{pos} vs {len(visit)}.\n"
        
        reduce_vars = []
        reduce_forward = []
        reduce_backward = []
        for iv in self[k].reduce_axis():
            var = tvm.tir.Var(iv.var.name, "int32")
            reduce_vars.append(var)
            reduce_forward.append(iv.var)
            reduce_backward.append(var)
        
        ret = _ffi_api.LayerStateTransform(self, k, spatial_vars, spatial_forward,
                                            spatial_backward, reduce_vars,
                                            reduce_forward, reduce_backward,
                                            1 if explicit else 0)
        return ret.output(0)
        

    def explicit_fold(self, k, axis, factor):
        return self._fold(k, axis, factor, True)

    def implicit_fold(self, k, axis, factor):
        return self._fold(k, axis, factor, False)
    
    def explicit_shuffle(self, k, *axes):
        return self._shuffle(k, *axes, explicit=True)

    def implicit_shuffle(self, k, *axes):
        return self._shuffle(k, *axes, explicit=False)
