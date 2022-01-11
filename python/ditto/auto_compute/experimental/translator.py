import string
from collections import OrderedDict, ChainMap

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from state import *
from utils import toposort


# maybe pad tensors depending on its index expression
def pad_tensor(tensor: Tensor, taccess: TensorAccess):
    # morphable tensors should never be padded
    # since only tensors affected by the sliding-window primitive require padding
    # and these tensors are set to non-morphable after the transform
    if tensor.morphable:
        pad_fn = lambda x: x
        pad_fn.config = None
        return tensor, pad_fn
    
    padded_shape = list()  # tensor shape after padding
    pth_pad_cfg = list()  # pytorch padding parameters
    idx_exprs = taccess.md_idx_expr.idx_exprs
    
    # 1. calculate the padding size along each dimension
    # padding area is the difference between the index range and the dimension
    # e.g. tensor dim: [0, 32], index range: [-2, 34] -> padding: [-2, -1] + [33, 34]
    for dim, idx in zip(tensor.shape, idx_exprs):
        # idx: 3*i + 4*j + 2 -> coeff_map: {i: 3, j: 4}, const_term: 2
        minv, maxv = idx.const_term, idx.const_term
        for it, coeff in idx.coeff_map.items():
            if coeff >= 0:
                minv += it.range.min * coeff
                maxv += it.range.max * coeff
            else: 
                minv += it.range.max * coeff
                maxv += it.range.min * coeff
        
        padding = [max(0, -minv), max(0, maxv-dim+1)]
        pth_pad_cfg.append(padding)
        padded_shape.append(dim + sum(padding))
    
    # 2. create the padded tensor
    padded_tensor = tensor.copy()
    padded_tensor.shape = padded_shape
    
    # 3. generate the padding function
    # which will be applied to realized pytorch tensors later
    # F.pad: The padding size is described starting from the last dimension and moving forward
    pth_pad_cfg = sum(reversed(pth_pad_cfg), list())
    padding_fn = lambda x: F.pad(x, pth_pad_cfg, mode='constant', value=0)
    padding_fn.config = {
        'pad': pth_pad_cfg,
        'mode': 'constant', 
        'value': 0,
    }

    return padded_tensor, padding_fn


# generate the strided view of a tensor according to its indices
# e.g. tensor[i+j] (affine idx) -> strided-view[i, j] (indices are DAVs/distinct atomic variables)
def gen_strided_view(tensor: Tensor, taccess: TensorAccess):
    idx_expr = taccess.md_idx_expr
    iters = list(idx_expr.iters)
    
    # 1. calculate size of the strided view
    sview_indices = iters  # indices of the strided view (DAVs)
    view_size = [it.range.extent for it in sview_indices]

    # 2. calculate strides of the strided view
    orig_stride = tensor.stride
    view_stride = list()
    for it in iters:
        stride = 0
        for s, idx in zip(orig_stride, idx_expr.idx_exprs):
            coeff = idx.coeff_map.get(it, 0)
            stride += coeff * s
        view_stride.append(stride)
    
    # 3. calculate the offset of the strided view
    # i.e. the offset of strided-view[0, ..., 0] (all itervars -> 0) in the storage
    offsets = [idx.const_term for idx in idx_expr.idx_exprs]
    view_offset = np.dot(offsets, orig_stride)

    # 4. generate the strided-view function
    # which will be applied to realized pytorch tensors later
    strided_view_fn = lambda tensor: tensor.as_strided(view_size, view_stride, view_offset)
    strided_view_fn.config = {
        'size': view_size,
        'stride': view_stride,
        'storage_offset': view_offset,
    }
    
    return sview_indices, strided_view_fn


class StageEinsum(nn.Module):
    def __init__(self, endpoints_config: dict, einsum_eq: str):
        super().__init__()
        self.endpoints_config = endpoints_config
        self.einsum_eq = einsum_eq
        self.params, self.tname_to_params = self._build_weights()

    @classmethod
    def from_compute_expr(cls, compute_expr: ComputeExpr):
        # 1. collect information for inputs and outputs
        inputs_sview_indices = list()  # indices of each strided view, used in einsum generation
        endpoints_config = dict()  # TODO: define a new class

        # 1.1 collect input infomation
        endpoints_config['inputs'] = list()
        for tacc in compute_expr.operands:
            # apply padding and as_strided
            padded_ts, padding_fn = pad_tensor(tacc.tensor, tacc)
            sview_indices, strided_view_fn = gen_strided_view(padded_ts, tacc)
            
            inputs_sview_indices.append(sview_indices)
            endpoints_config['inputs'].append({
                'name': padded_ts.name,
                'shape': padded_ts.shape,
                'sview_fn': strided_view_fn,
                'pad_fn': padding_fn,
                'is_weight': tacc.morphable and len(tacc.tensor.accesses) == 1,
            })

        # 1.2 collect output information
        # output tensor needs no padding, as sliding-window only applies to inputs
        output_tensor = compute_expr.output.tensor
        __, outp_sview_fn = gen_strided_view(output_tensor, compute_expr.output)
        endpoints_config['output'] = {
            'name': output_tensor.name,
            'shape': output_tensor.shape,
            'sview_fn': outp_sview_fn,
        }

        # 2. generate einsum equation
        outp_iters = list(compute_expr.output.md_idx_expr.iters)
        all_iters = list(compute_expr.iters)

        # 2.1 assign a new name to each iterator
        # subscripts in einsum must be letters from [a-zA-Z]
        legal_subscripts = string.ascii_lowercase + string.ascii_uppercase
        assert len(all_iters) <= len(legal_subscripts)
        new_iter_names = dict(zip(all_iters, legal_subscripts))

        # 2.2 assemble the final einsum equation
        operand_indices = [
            ''.join(new_iter_names[it] for it in sview_indices)
            for sview_indices in inputs_sview_indices
        ]
        output_index = ''.join(new_iter_names[it] for it in outp_iters)
        einsum_eq = '->'.join([','.join(operand_indices), output_index])

        return cls(endpoints_config, einsum_eq)

    @property
    def input_cfg(self):
        return {
            inp_cfg['name']: inp_cfg['shape']
            for inp_cfg in self.endpoints_config['inputs'] if not inp_cfg['is_weight']
        }

    @property
    def output_cfg(self):
        out_cfg = self.endpoints_config['output']
        return {
            'name': out_cfg['name'],
            'shape': out_cfg['shape'],
        }

    @property
    def num_inputs(self): return len(self.input_cfg)

    def _build_weights(self):
        ordered_params = list()
        tname_to_params = OrderedDict()  # tensor name -> weight

        for inp_cfg in self.endpoints_config['inputs']:
            if not inp_cfg['is_weight']: continue
            
            param_name = 'W%d' % len(ordered_params)
            param = Parameter(torch.randn(*inp_cfg['shape']), True)  # TODO: choose a proper init scheme
            self.register_parameter(param_name, param)
            
            ordered_params.append(param)
            tname_to_params[inp_cfg['name']] = param
        
        return ordered_params, tname_to_params

    def _prepare_inputs(self, inps: "dict[str, torch.Tensor]") -> "list[torch.Tensor]":
        inp_wgt = {**inps, **self.tname_to_params}
        
        prepared_inps = list()
        for inp_cfg in self.endpoints_config['inputs']: 
            x = inp_wgt[inp_cfg['name']]
            x = inp_cfg['pad_fn'](x)
            x = inp_cfg['sview_fn'](x)
            prepared_inps.append(x)

        return prepared_inps

    def _postproc_output(self, outp: torch.Tensor) -> torch.Tensor:
        out_cfg = self.endpoints_config['output']
        outp_buf = outp.new_zeros(out_cfg['shape'])
        outp_buf = out_cfg['sview_fn'](outp_buf)
        outp_buf.copy_(outp)
        return outp_buf
    
    def forward(self, **inps):
        inputs = self._prepare_inputs(inps)
        output = torch.einsum(self.einsum_eq, *inputs)
        output = self._postproc_output(output)
        return output

    def to_json(self):
        endp_cfg = self.endpoints_config.copy()
        
        for inp_cfg in endp_cfg['inputs']: 
            inp_cfg['sview_fn'] = inp_cfg['sview_fn'].config
            inp_cfg['pad_fn'] = inp_cfg['pad_fn'].config
        
        outp_cfg = endp_cfg['output']
        outp_cfg['sview_fn'] = outp_cfg['sview_fn'].config

        return {
            **endp_cfg,
            'einsum': self.einsum_eq,
        }

    @classmethod
    def from_json(cls, json):
        json = json.copy()
        einsum_eq = json.pop('einsum')
        endp_cfg = json

        for inp_cfg in endp_cfg['inputs']: 
            inp_cfg['sview_fn'] = lambda x: x.as_strided(**inp_cfg['sview_fn'].config)
            pad_fn_cfg = inp_cfg['pad_fn'].config
            if not pad_fn_cfg:
                inp_cfg['pad_fn'] = lambda x: x
            else:
                inp_cfg['pad_fn'] = lambda x, cfg=pad_fn_cfg: F.pad(x, **cfg)

        outp_cfg = endp_cfg['output']
        outp_cfg['sview_fn'] = lambda x: x.as_strided(**outp_cfg['sview_fn'].config)

        return cls(endp_cfg, einsum_eq)


class StateEinsum(nn.Module):
    def __init__(self, sorted_mods: "list[StageEinsum]", tensor_cfg, endpoints_cfg):
        super().__init__()
        
        self.sorted_mods = sorted_mods
        for i, mod in enumerate(sorted_mods):
            setattr(self, 'stage%d' % i, mod)

        self.tensor_cfg = tensor_cfg
        self.endpoints_cfg = endpoints_cfg

    @classmethod
    def from_state(cls, state: State):
        # 1. build StageEinsum module for each stage
        stage_to_mod = OrderedDict({
            stage: StageEinsum.from_compute_expr(stage.compute_expr)
            for stage in state.stages
        })

        # 2. toposort stages and corresponding modules
        def get_consumers(stage):
            output = stage.compute_expr.output
            consumers = [
                acc.compute_expr.stage 
                for acc in output.tensor.accesses
                if acc is not output
            ]
            return consumers

        sorted_stages = toposort(state.stages, edges=OrderedDict({
            stage: get_consumers(stage)
            for stage in state.stages
        }))
        sorted_mods = [stage_to_mod[stage] for stage in sorted_stages]
        
        # 3. collect information for tensors
        # 3.1 information for all tensors
        tensor_cfg = {
            tensor.name: tensor.shape
            for tensor in state.tensors
        }

        # 3.2 find inputs and outputs of the subgraph
        input_tnames, output_tnames = list(), list()
        for stage in state.stages:
            expr = stage.compute_expr
            
            for access in [expr.output, *expr.operands]:
                if access.morphable: continue
                tensor = access.tensor
                
                n_producers, n_consumers = 0, 0
                for acc in tensor.accesses:
                    if acc.compute_expr.output.tensor is tensor:
                        n_producers += 1
                    else: 
                        n_consumers += 1
                
                if n_producers == 0:
                    input_tnames.append(tensor.name)
                
                if n_consumers == 0:
                    output_tnames.append(tensor.name)
        
        endpoints_cfg = {
            'input_names': input_tnames,
            'output_names': output_tnames,
        }

        return cls(sorted_mods, tensor_cfg, endpoints_cfg)
    
    @property
    def input_cfg(self):
        return {
            inp_name: self.tensor_cfg[inp_name]
            for inp_name in self.endpoints_cfg['input_names']
        }

    @property
    def output_cfg(self):
        return {
            outp_name: self.tensor_cfg[outp_name]
            for outp_name in self.endpoints_cfg['output_names']
        }

    @property
    def params(self):
        return sum([mod.params for mod in self.sorted_mods], list())

    @property
    def tname_to_params(self):
        return ChainMap(*[mod.tname_to_params for mod in self.sorted_mods])

    def forward(self, **inps):
        buffer_map = dict.fromkeys(self.tensor_cfg.keys(), None)
        buffer_map.update(self.tname_to_params)
        buffer_map.update(inps)

        for mod in self.sorted_mods:
            inputs = {k: buffer_map[k] for k in mod.input_cfg.keys()}
            output = mod(**inputs)
            buffer_map[mod.output_cfg['name']] = output

        outputs = {
            k: buffer_map[k]
            for k in self.output_cfg.keys()
        }
        return outputs

    def to_json(self):
        return {
            'modules': [mod.to_json() for mod in self.sorted_mods], 
            'tensors': self.tensor_cfg,
            'endpoints': self.endpoints_cfg,
        }

    @classmethod
    def from_json(cls, json):
        sorted_mods = [StageEinsum.from_json(mod_json) for mod_json in json['modules']]
        tensor_cfg = json['tensors']
        endp_cfg = json['endpoints']
        return cls(sorted_mods, tensor_cfg, endp_cfg)
