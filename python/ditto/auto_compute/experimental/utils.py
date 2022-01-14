from contextlib import contextmanager
from typing import Any, Tuple, NamedTuple, Optional, Dict

import torch
import torch.fx
import torch.nn as nn
from torch.fx.node import Node, map_aggregate


class DisjointSet:
    def __init__(self, elements, parent):
        self.elements = elements
        self.parent = parent

    def find(self, item):
        if self.parent[item] == item:
            return item
        else:
            res = self.find(self.parent[item])
            self.parent[item] = res
            return res

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        self.parent[root1] = root2


def disjoint_set_union(groups: "list[tuple]"):
    elements = list(set(sum(map(list, groups), list())))
    parents = dict(zip(elements, elements))
    ds = DisjointSet(elements, parents)
    
    for grp in groups:
        for elem in grp[1:]:
            ds.union(grp[0], elem)

    root_to_group = dict()
    for elem in elements:
        root = ds.find(elem)
        if root not in root_to_group:
            root_to_group[root] = list()
        root_to_group[root].append(elem)

    groups = list(root_to_group.values())
    return groups


def toposort(vertices: list, edges):
    def _toposort(v, visited, sorted_vs):
        visited[v] = True
        if v in edges:
            for u in edges[v]:
                if visited[u]: continue
                _toposort(u, visited, sorted_vs)
        sorted_vs.append(v)

    visited = dict.fromkeys(vertices, False)
    sorted_vs = list()

    for v in vertices:
        if visited[v]: continue
        _toposort(v, visited, sorted_vs)

    return sorted_vs[::-1]


def same_padding(length, kernel_size, stride):
    if length % stride == 0:
        tot_pad = max(kernel_size - stride, 0)
    else:
        tot_pad = max(kernel_size - (length % stride), 0)

    pad_left = tot_pad // 2
    pad_right = tot_pad - pad_left

    return pad_left, pad_right


""" 
extract metadata of intermediate tensors
source: https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py
 """

class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    requires_grad : bool
    stride : Tuple[int]
    memory_format : Optional[torch.memory_format]

    # Quantization metadata
    is_quantized : bool
    qparams: Dict[str, Any]


def _extract_tensor_metadata(result : torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)


class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Args:
         module (GraphModule): The module to be executed

    """
    def run_node(self, n : Node) -> Any:
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)


def extract_tensor_metadata(model: nn.Module, sample_input: torch.Tensor):
    graph_mod = torch.fx.symbolic_trace(model)
    ShapeProp(graph_mod).propagate(sample_input)

    tensor_meta = list()
    for node in graph_mod.graph.nodes:
        if 'tensor_meta' not in node.meta: continue
        meta = node.meta['tensor_meta']
        tensor_meta.append({
            'name': node.name, 
            'dtype': meta.dtype,
            'shape': meta.shape,
        })
    
    return tensor_meta


def test_shape_prop():
    """
    Example:
    
    In this example, we record the shape
    and data type of a module given
    an example input ``torch.randn(50, D_in)``.
    We print the name, shape and dtype of each node.
    
    The output of this code is:

    x torch.float32 torch.Size([50, 1000])
    linear1 torch.float32 torch.Size([50, 100])
    clamp_1 torch.float32 torch.Size([50, 100])
    linear2 torch.float32 torch.Size([50, 10])
    output torch.float32 torch.Size([50, 10])
    """


    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)
        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    D_in, H, D_out = 1000, 100, 10
    model = TwoLayerNet(D_in, H, D_out)
    gm = torch.fx.symbolic_trace(model)
    sample_input = torch.randn(50, D_in)
    ShapeProp(gm).propagate(sample_input)

    for node in gm.graph.nodes:
        print(node.name, node.meta['tensor_meta'].dtype, node.meta['tensor_meta'].shape)


""" Utils for Analyzing PyTorch Models """


@contextmanager
def set_model_eval(model):
    is_training = model.training
    model = model.eval()

    try:
        yield
    finally:
        model = model.train() if is_training else model.eval()


@contextmanager
def record_io_shapes(model, sample_inputs, mods=None):
    io_shapes = dict()
    
    def _record_io_shapes(m, i, o):
        io_shapes[m] = ([ii.shape for ii in i], o.shape)
        m.input_shapes = [ii.shape for ii in i]
        m.output_shape = o.shape
    
    hook_handles = list()
    for mod in (mods or model.modules()):
        handle = mod.register_forward_hook(_record_io_shapes)
        hook_handles.append(handle)

    with set_model_eval(model):
        with torch.no_grad():
            model(*sample_inputs)

    try:
        yield io_shapes.copy()
    finally:
        for handle in hook_handles:
            handle.remove()

        for mod in io_shapes.keys():
            del mod.input_shapes, mod.output_shape


@contextmanager
def record_mod_qualnames(model):
    qualnames = dict()

    def _set_qualname(mod, prefix):
        mod.qualname = prefix
        qualnames[mod] = prefix
        for name, submod in mod.named_children():
            _set_qualname(submod, f'{prefix}.{name}')
    
    _set_qualname(model, prefix='')

    try:
        yield qualnames.copy()
    finally:
        for mod in qualnames.keys():
            del mod.qualname


@contextmanager
def record_mod_parent(model):
    parents = dict()

    def _set_parent(mod):
        for submod in mod.children():
            parents[submod] = mod
            assert not hasattr(submod, 'parent')  # only one parent
            submod.parent = [mod]  # avoid submodule registeration
            _set_parent(submod)
    
    parents[model] = None
    model.parent = [None]
    _set_parent(model)

    try:
        yield parents.copy()
    finally:
        for mod in parents.keys():
            del mod.parent


def replace_module(parent, mod, new_mod):
    mod_name = None
    for name, submod in parent.named_children():
        if submod is mod:
            mod_name = name
    
    if mod_name is None:
        raise RuntimeError

    setattr(parent, mod_name, new_mod)


if __name__ == "__main__":
    vertices = ['a', 'b', 'c', 'd']
    edges = {
        'a': ['b', 'c'],
        'b': ['d'],
        'c': ['d'],
        'd': [],
    }
    print(toposort(vertices, edges))
