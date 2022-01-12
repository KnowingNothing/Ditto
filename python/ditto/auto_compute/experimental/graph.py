from expr import *
from state import *


# TODO: support other basic operators such as add and concat
class Subgraph:
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

    """ user interface """
    
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


class Graph:
    pass
