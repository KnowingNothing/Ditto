from expr import *
from state import *


# TODO: support specifying output shape of a subgraph
# TODO: support other basic operators such as add and concat
class Graph:
    def __init__(self):
        self.tensors = list()
        self.weights = list()
        self.stages = list()
        self._state = None

    @property
    def state(self):
        if self._state is None:
            new_state = State(self.stages)
            self._state = new_state
        return self._state

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
    
    def new_tensor(self, name, shape: list):
        t = self._new_tensor(name, shape, morphable=False)
        return t

    # build stage: outputs = inputs @ W
    def linear_map(self, inputs: Tensor, outputs: Tensor, weight_name=None):
        self._check_tensor_exist(inputs, outputs)

        # if inputs is an intermediate output, set inputs to morphble
        if len(inputs.accesses) > 0:
            inputs.set_morphable(True)

        weight_shape = outputs.shape + inputs.shape
        weight_name = weight_name or f'weight_{len(self.weights)}'
        weight = self._new_weight(weight_name, weight_shape)

        stage_id = len(self.stages)
        inputs_iters = self._build_tensor_iters(inputs, True, f's{stage_id}_i')  # reduce iters
        outputs_iters = self._build_tensor_iters(outputs, False, f's{stage_id}_o')  # spatial iters
        all_iters = outputs_iters + inputs_iters

        inputs_access = TensorAccess.new_init(inputs, inputs_iters)
        weight_access = TensorAccess.new_init(weight, all_iters)
        outputs_access = TensorAccess.new_init(outputs, outputs_iters)
        fc_expr = ComputeExpr(outputs_access, operands=[inputs_access, weight_access])

        new_stage = Stage(all_iters, fc_expr)
        self.stages.append(new_stage)

        return weight

    # build stage: outputs = outer(input_a, input_b)
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


# TODO: move compute definition to utils.py
def test_graph1():
    graph = Graph()
    X = graph.new_tensor('X', [32, 28, 28])
    Y = graph.new_tensor('Y', [64, 14, 14])
    W = graph.linear_map(X, Y, 'W')
    
    print('Compute Definition:')
    print(graph.state.stages[0].compute_expr)
    
    return graph.state


def test_graph2():
    graph = Graph()
    X1 = graph.new_tensor('X1', [32, 28, 28])
    Y1 = graph.new_tensor('Y1', [64, 14, 14])
    W1 = graph.linear_map(X1, Y1, 'W1')
    X2 = graph.new_tensor('X2', [32, 28, 28])
    Y2 = graph.new_tensor('Y2', [64, 14, 14])
    W2 = graph.linear_map(X2, Y2, 'W2')
    O1 = graph.outer_prod(Y1, Y2, 'O1')
    O = graph.new_tensor('O', [64, 7, 7])
    W3 = graph.linear_map(O1, O, 'W3')

    state = graph.state
    
    print('Compute Definition:')
    print(state.stages[0].compute_expr)
    print(state.stages[1].compute_expr)
    print(state.stages[2].compute_expr)
    print(state.stages[3].compute_expr)
    
    return state


if __name__ == '__main__':
    test_graph1()
    test_graph2()
