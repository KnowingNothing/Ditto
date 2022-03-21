import tvm._ffi
import tvm
from tvm.te import tensor
from . import _ffi_api
from tvm.runtime import Object
from .graph import LayerTensor, Layer, Graph


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

    def transform(
        self, spatial_forward, fspatial_backward, reduce_forward, freduce_backward
    ):
        def _make_vars(forward, fcompute):
            code = fcompute.__code__
            names = code.co_varnames
            if not (len(forward) == len(names)):
                raise ValueError(
                    f"transform var number mismatch {len(forward)} vs {len(names)}.\n"
                )
            vvars = [tvm.tir.Var(name, "int32") for name in names]
            return vvars, fcompute(*vvars)

        spatial_vars, spatial_backward = _make_vars(spatial_forward, fspatial_backward)
        reduce_vars, reduce_backward = _make_vars(reduce_forward, freduce_backward)
        return _ffi_api.OpStateTransform(
            self,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
        )


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

    def get_current_ops(self):
        return _ffi_api.LayerStateGetCurrentOps(self)

    def get_op_id_in_current_ops(self, op):
        all_ops = self.get_current_ops()
        for i, p in enumerate(all_ops):
            if op == p:
                return i
        raise ValueError(f"op: {op} is not part of this layer: {self}.\n")

    def transform(
        self,
        k,
        spatial_forward,
        fspatial_backward,
        reduce_forward,
        freduce_backward,
        explicit_transform=True,
    ):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")

        def _make_vars(forward, fcompute):
            code = fcompute.__code__
            names = code.co_varnames
            if not (len(forward) == len(names)):
                raise ValueError(
                    f"transform var number mismatch {len(forward)} vs {len(names)}.\n"
                )
            vvars = [tvm.tir.Var(name, "int32") for name in names]
            return vvars, fcompute(*vvars)

        spatial_vars, spatial_backward = _make_vars(spatial_forward, fspatial_backward)
        reduce_vars, reduce_backward = _make_vars(reduce_forward, freduce_backward)
        ret = _ffi_api.LayerStateTransform(
            self,
            k,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
            1 if explicit_transform else 0,
        )
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
                    forward.extend([iv.var // factor, iv.var % factor])
                    backward.append(outer * factor + inner)
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
        _inner(reduce_vars, reduce_forward, reduce_backward, self[k].reduce_axis())

        ret = _ffi_api.LayerStateTransform(
            self,
            k,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
            1 if explicit else 0,
        )
        return ret.output(0)

    def _unfold(self, k, *axis, explicit=True):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")

        # check all spatial or reduce
        iter_type = None
        for iv in axis:
            if iter_type is None:
                iter_type = iv.iter_type
            else:
                if iter_type != iv.iter_type:
                    raise ValueError("Can't unfold different types of axis.\n")

        def _inner(all_axis, vars, forward, backward):
            pos_dict = {}
            for i, iv in enumerate(all_axis):
                pos_dict[iv] = i

            # check valid input arguments
            curr_id = -1
            for iv in axis:
                assert iv in pos_dict
                next_id = pos_dict[iv]
                if curr_id < 0:
                    curr_id = next_id
                else:
                    if curr_id + 1 != next_id:
                        raise ValueError("Must unfold adjacent axis.\n")
                    curr_id = next_id
            num_axis = len(axis)
            start_id = curr_id - num_axis + 1
            end_id = curr_id + 1

            fused_var = None
            factors = [1 for i in range(num_axis + 1)]

            for i, iv in enumerate(all_axis):
                if i < start_id or i >= end_id:
                    var = tvm.tir.Var(iv.var.name, "int32")
                    vars.append(var)
                    forward.append(iv.var)
                    backward.append(var)
                elif i == start_id:
                    names = []
                    unfold_expr = 0
                    for siv in all_axis[i : i + num_axis]:
                        names.append(siv.var.name)
                        unfold_expr = unfold_expr * siv.dom.extent + siv.var
                    for j, siv in enumerate(reversed(all_axis[i : i + num_axis])):
                        factors[num_axis - j - 1] = (
                            factors[num_axis - j] * siv.dom.extent
                        )
                    name = ".".join(names) + ".unfold"
                    var = tvm.tir.Var(name, "int32")
                    fused_var = var
                    vars.append(var)
                    forward.append(unfold_expr)
                    backward.append(fused_var // factors[i - start_id + 1])
                else:
                    backward.append(
                        fused_var % factors[i - start_id] // factors[i - start_id + 1]
                    )

        spatial_vars = []
        spatial_forward = []
        spatial_backward = []
        reduce_vars = []
        reduce_forward = []
        reduce_backward = []
        if iter_type == 0:
            # spatial
            all_axis = self[k].axis()
            _inner(all_axis, spatial_vars, spatial_forward, spatial_backward)
            for iv in self[k].reduce_axis():
                var = tvm.tir.Var(iv.var.name, "int32")
                reduce_vars.append(var)
                reduce_forward.append(iv.var)
                reduce_backward.append(var)
        elif iter_type == 2:
            # reduce
            all_axis = self[k].reduce_axis()
            _inner(all_axis, reduce_vars, reduce_forward, reduce_backward)
            for iv in self[k].axis():
                var = tvm.tir.Var(iv.var.name, "int32")
                spatial_vars.append(var)
                spatial_forward.append(iv.var)
                spatial_backward.append(var)
        else:
            raise ValueError(f"Unsupported iter var type: {iter_type}.\n")
        ret = _ffi_api.LayerStateTransform(
            self,
            k,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
            1 if explicit else 0,
        )
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

        ret = _ffi_api.LayerStateTransform(
            self,
            k,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
            1 if explicit else 0,
        )
        return ret.output(0)

    def _eliminate(self, k, axis, explicit=True):
        if isinstance(k, tensor.Tensor):
            k = k.op
        if not isinstance(k, tensor.Operation):
            raise ValueError("Expect state key to be Tensor or Operation")

        exist = False
        for iv in self[k].axis():
            if iv == axis:
                exist = True
                if len(self[k].axis()) == 1:
                    raise ValueError(
                        "Can't eliminate the only spatial dimension of the compute"
                    )
                break
        for iv in self[k].reduce_axis():
            if iv == axis:
                exist = True
                if len(self[k].reduce_axis()) == 1:
                    raise ValueError(
                        "Can't eliminate the only reduce dimension of the tensor"
                    )
                break

        if not exist:
            raise ValueError(f"The axis {axis} is not part of the op {k}.\n")

        def _inner(all_axis, vars, forward, backward):
            for iv in all_axis:
                if iv == axis:
                    backward.append(0)
                else:
                    var = tvm.tir.Var(iv.var.name, "int32")
                    vars.append(var)
                    forward.append(iv.var)
                    backward.append(var)

        spatial_vars = []
        spatial_forward = []
        spatial_backward = []
        _inner(self[k].axis(), spatial_vars, spatial_forward, spatial_backward)

        reduce_vars = []
        reduce_forward = []
        reduce_backward = []
        _inner(self[k].reduce_axis(), reduce_vars, reduce_forward, reduce_backward)

        ret = _ffi_api.LayerStateTransform(
            self,
            k,
            spatial_vars,
            spatial_forward,
            spatial_backward,
            reduce_vars,
            reduce_forward,
            reduce_backward,
            1 if explicit else 0,
        )
        return ret.output(0)

    def explicit_fold(self, k, axis, factor):
        return self._fold(k, axis, factor, True)

    def implicit_fold(self, k, axis, factor):
        return self._fold(k, axis, factor, False)

    def explicit_unfold(self, k, *axis):
        return self._unfold(k, *axis, explicit=True)

    def implicit_unfold(self, k, *axis):
        return self._unfold(k, *axis, explicit=False)

    def explicit_shuffle(self, k, *axes):
        return self._shuffle(k, *axes, explicit=True)

    def implicit_shuffle(self, k, *axes):
        return self._shuffle(k, *axes, explicit=False)

    def explicit_eliminate(self, k, axis):
        return self._eliminate(k, axis, explicit=True)

    def implicit_eliminate(self, k, axis):
        return self._eliminate(k, axis, explicit=False)


@tvm._ffi.register_object("ditto.auto_compute.GraphState")
class GraphState(Object):
    """GraphState object"""

    def __getitem__(self, k):
        if isinstance(k, LayerTensor):
            k = k.layer
        if not isinstance(k, Layer):
            raise ValueError("Expect state key to be LayerTensor or Layer")
        return _ffi_api.GraphStateGetLayerState(self, k)

    def make_compute(self, layer_inputs):
        return _ffi_api.GraphStateMakeCompute(self, layer_inputs)

    def get_current_layers(self):
        return _ffi_api.GraphStateGetCurrentLayers(self)

    def normalize_partition_layer(self, layer, modify=True):
        return _ffi_api.GraphStateNormalizePartitionLayer(self, layer, modify)

    def fuse_layer(self, front, back, modify=True):
        return _ffi_api.GraphStateFuseLayer(self, front, back, modify)


def create_op_state(op):
    """Make op state"""
    return _ffi_api.CreateOpState(op)


def create_layer_state(layer):
    """Make the Layer State"""
    return _ffi_api.CreateLayerState(layer)


def create_graph_state(graph):
    """Make the Graph State"""
    return _ffi_api.CreateGraphState(graph)


def find_convex_layers(front, back):
    return _ffi_api.FindConvexSet(front, back)
