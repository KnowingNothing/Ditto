from typing import Callable
from enum import Enum, auto

import torch
import torch.nn as nn

from expr import *
from state import *
from translator import StateEinsum
from utils import record_io_shapes, record_mod_qualnames, record_mod_parent, replace_module


# TODO: generate subgraph from the original module automatically
# TODO: support other basic operators such as add and concat
# Utility class for defining init state
class InitStateBuilder:
    def __init__(self):
        self.tensors: list[Tensor] = list()
        self.weights: list[Tensor] = list()
        self.stages: list[Stage] = list()
        self._state: Optional[State] = None

    @property
    def state(self):
        if self._state is None:
            new_state = State(self.stages)
            self._state = new_state
        return self._state

    @property
    def inputs(self) -> "list[Tensor]":
        inputs = list()
        for stage in self.state.stages: 
            for acc in stage.compute_expr.operands:
                if len(acc.tensor.accesses) > 1: continue  # not leaf nodes
                if acc.morphable: continue  # weights
                inputs.append(acc.tensor)
        return inputs

    @property
    def outputs(self) -> "list[Tensor]":
        outputs = list()
        for stage in self.state.stages:
            outp = stage.compute_expr.output.tensor
            if len(outp.accesses) > 1: continue  # not leaf nodes
            outputs.append(outp)
        return outputs

    def _new_tensor(self, name, shape, morphable=False):
        t = Tensor(name, shape, morphable)
        self.tensors.append(t)
        return t

    def _new_weight(self, name, shape):
        t = self._new_tensor(name, shape, morphable=True)
        self.weights.append(t)
        return t

    def _check_tensor_exist(self, *tensors):
        for t in tensors: 
            if t not in self.tensors:
                raise ValueError(f"Tensor {t.name} does not exist.")

    # TODO: support named dimensions
    def _build_tensor_iters(self, tensor: Tensor, reduce: bool, prefix=None) -> "list[Iter]":
        iters = list()
        for idx, dim in enumerate(tensor.shape):
            iter_name = f'it{idx}'
            if prefix is not None: 
                iter_name = f'{prefix}_{iter_name}'
            new_iter = Iter(iter_name, Range(0, dim), reduce)
            iters.append(new_iter)
        return iters

    """ user interface for construct initial subgraph """
    
    def new_input(self, shape: list, name):
        t = self._new_tensor(name, shape, morphable=False)
        return t

    def linear_map(self, inputs: Tensor, outputs_shape: "list[int]", outputs_name=None, weight_name=None):
        self._check_tensor_exist(inputs)

        # if inputs is an intermediate output, set inputs to morphble
        if len(inputs.accesses) > 0:
            inputs.set_morphable(True)

        stage_id = len(self.stages)
        outputs_name = outputs_name or f'output_{stage_id}'
        outputs = self._new_tensor(outputs_name, outputs_shape, morphable=False)

        weight_shape = outputs.shape + inputs.shape
        weight_name = weight_name or f'weight_{len(self.weights)}'
        weight = self._new_weight(weight_name, weight_shape)

        inputs_iters = self._build_tensor_iters(inputs, True, f's{stage_id}_i')  # reduce iters
        outputs_iters = self._build_tensor_iters(outputs, False, f's{stage_id}_o')  # spatial iters
        all_iters = outputs_iters + inputs_iters

        inputs_access = TensorAccess.new_init(inputs, inputs_iters)
        weight_access = TensorAccess.new_init(weight, all_iters)
        outputs_access = TensorAccess.new_init(outputs, outputs_iters)
        linear_expr = ComputeExpr(outputs_access, operands=[inputs_access, weight_access])

        new_stage = Stage(all_iters, linear_expr)
        self.stages.append(new_stage)

        return outputs, weight

    # TODO: deprecated -> bilinear_map, to be removed
    # build stage: outputs = outer(input_a, input_b)
    # bilinear-map = outer-prod + linear-map
    def outer_prod(self, input_a: Tensor, input_b: Tensor, outputs_name=None):
        self._check_tensor_exist(input_a, input_b)

        # if inputs is an intermediate output, set inputs to morphble
        for ts in [input_a, input_b]:
            if len(ts.accesses) > 0: 
                ts.set_morphable(True)

        stage_id = len(self.stages)
        outputs_shape = input_a.shape + input_b.shape
        outputs_name = outputs_name or f'output_{stage_id}'
        outputs = self._new_tensor(outputs_name, outputs_shape, morphable=False)

        inp_a_iters = self._build_tensor_iters(input_a, False, f's{stage_id}_a')  # spatial iters
        inp_b_iters = self._build_tensor_iters(input_b, False, f's{stage_id}_b')  # spatial iters
        all_iters = inp_b_iters + inp_a_iters

        inp_a_access = TensorAccess.new_init(input_a, inp_a_iters)
        inp_b_access = TensorAccess.new_init(input_b, inp_b_iters)
        outputs_access = TensorAccess.new_init(outputs, all_iters)
        out_prod_expr = ComputeExpr(outputs_access, operands=[inp_a_access, inp_b_access])

        new_stage = Stage(all_iters, out_prod_expr)
        self.stages.append(new_stage)
        
        return outputs

    def bilinear_map(self, input_a: Tensor, input_b: Tensor, outputs_shape: "list[int]", outputs_name=None, weight_name=None):
        self._check_tensor_exist(input_a, input_b)

        # if inputs is an intermediate output, set inputs to morphble
        for ts in [input_a, input_b]:
            if len(ts.accesses) > 0: 
                ts.set_morphable(True)

        stage_id = len(self.stages)
        outputs_name = outputs_name or f'output_{stage_id}'
        outputs = self._new_tensor(outputs_name, outputs_shape, morphable=False)
        
        weight_shape = outputs.shape + input_a.shape + input_b.shape
        weight_name = weight_name or f'weight_{len(self.weights)}'
        weight = self._new_weight(weight_name, weight_shape)

        input_a_iters = self._build_tensor_iters(input_a, True, f's{stage_id}_ia')  # reduce iters
        input_b_iters = self._build_tensor_iters(input_b, True, f's{stage_id}_ib')  # reduce iters
        outputs_iters = self._build_tensor_iters(outputs, False, f's{stage_id}_o')  # spatial iters
        all_iters = outputs_iters + input_a_iters + input_b_iters

        input_a_access = TensorAccess.new_init(input_a, input_a_iters)
        input_b_access = TensorAccess.new_init(input_b, input_b_iters)
        weight_access = TensorAccess.new_init(weight, all_iters)
        outputs_access = TensorAccess.new_init(outputs, outputs_iters)
        bilinear_expr = ComputeExpr(outputs_access, operands=[input_a_access, input_b_access, weight_access])

        new_stage = Stage(all_iters, bilinear_expr)
        self.stages.append(new_stage)

        return outputs, weight

    def activation(self, inputs: Tensor, act_key='relu'):
        self._check_tensor_exist(inputs)
        stage = inputs.producer
        assert stage is not None
        stage.set_activation(act_key)
        return inputs


class MorphBlock(nn.Module):
    
    class ForwardMode(Enum):
        ORIGINAL = auto()
        GENERATED = auto()
        UNDEFINED = auto()

    def __init__(self, orig_mod: nn.Module, init_stt_bldr: InitStateBuilder):
        super().__init__()
        self.orig_mod = orig_mod  # original PyTorch module that we replace
        self.stt_bldr = init_stt_bldr  # initial state definition
        self.gen_mod: "Optional[StateEinsum]" = None  # generated Pytorch module
        self.fwd_mode = MorphBlock.ForwardMode.UNDEFINED 

    @property
    def init_state(self):
        return self.stt_bldr.state

    def set_forward_mode(self, mode: ForwardMode):
        self.mode = mode
    
    def set_generated_mod(self, gen_mod: StateEinsum):
        self.gen_mod = gen_mod

    def forward(self, *args, **kwargs):
        if self.mode is MorphBlock.ForwardMode.ORIGINAL:
            return self.orig_mod(*args, **kwargs)
        elif self.mode is MorphBlock.ForwardMode.GENERATED:
            if self.gen_mod is None:
                raise RuntimeError('You have not set the generated module.')
            return self.gen_mod(*args, **kwargs)
        else:
            raise RuntimeError("You have not set a forward mode.")


class ModelRewriter:
    def __init__(self, model):
        self.model = model

    def insert_morph_blocks(self, mods_to_replace, input_shapes, init_state_fn: Callable):
        sample_inputs = [torch.randn(shape) for shape in input_shapes]

        with record_mod_parent(self.model):
            with record_mod_qualnames(self.model):
                with record_io_shapes(self.model, sample_inputs, mods_to_replace):
                    for orig_mod in mods_to_replace:
                        assert isinstance(orig_mod.parent, list)
                        init_state_bldr = init_state_fn(
                            orig_mod, orig_mod.parent[0], orig_mod.qualname, 
                            orig_mod.input_shapes, orig_mod.output_shape
                        )
                        morph_block = MorphBlock(orig_mod, init_state_bldr)
                        replace_module(orig_mod.parent[0], orig_mod, morph_block)

    @property
    def morph_blocks(self) -> "list[MorphBlock]":
        morph_blocks = list()
        for n, m in self.model.named_modules():
            if isinstance(m, MorphBlock):
                morph_blocks.append(MorphBlock)
        return morph_blocks

    def set_forward_mode(self, fwd_mode: MorphBlock.ForwardMode):
        for mb in self.morph_blocks:
            mb.set_forward_mode(fwd_mode)

    def prune_morph_blocks(self):
        for mb in self.morph_blocks:
            self.prune_morph_block(mb)

    def prune_morph_block(self, mod: MorphBlock):
        assert mod in self.morph_blocks
        assert mod.gen_mod is not None
        with record_mod_parent(self.model):
            assert isinstance(mod.parent, list)
            replace_module(mod.parent[0], mod, mod.gen_mod)


def test_model_rewriter():
    from torchvision.models import resnet18
    
    model = resnet18(pretrained=False)
    mods_to_replace = [model.layer1, model.layer2, model.layer3, model.layer4]
    input_shapes = [[1, 3, 224, 224]]

    def gen_resnet18_blocks(mod, parent, qualname, input_shapes, output_shape):        
        assert len(input_shapes) == 1
        bs, ic, ires, ires = input_shapes[0]
        bs, oc, ores, ores = output_shape

        graph = InitStateBuilder()
        x = graph.new_input([ic, ires, ires], 'x')
        x = graph.linear_map(x, [oc, ores, ores], 'conv1', 'conv1_W')[0]
        x = graph.activation(x, act_key='relu')
        x = graph.linear_map(x, [oc, ores, ores], 'conv2', 'conv2_W')[0]
        x = graph.activation(x, act_key='relu')
        return graph

    rewriter = ModelRewriter(model)
    rewriter.insert_morph_blocks(mods_to_replace, input_shapes, gen_resnet18_blocks)
    print(model)


if __name__ == '__main__':
    test_model_rewriter()
