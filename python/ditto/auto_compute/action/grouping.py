import json
import numpy as np
from ditto.auto_compute.state import LayerState
from .action_base import Guarder, Actor, Action, ParameterSpace, Parameter
from .. import _ffi_api
from .record import TransformApplier, TransformRecord, ActionRecord
from .utils import get_factor_lst


def find_grouping_pattern(op):
    return _ffi_api.FindGroupingPattern(op)


class GroupingGuarder(Guarder):
    @staticmethod
    def applicable(layer_state, op_id):
        all_ops = layer_state.get_current_ops()
        cur_op = all_ops[op_id]
        patterns = find_grouping_pattern(layer_state[cur_op].op)
        return (len(patterns) > 0), patterns


class GroupingActor(Actor):
    @staticmethod
    def apply(layer_state, op_id, pattern, parameter):
        # pattern contains IntImm, not int
        data_id, weight_id = pattern.tensor_ids
        siv_id, riv_id = pattern.iter_ids_array[0]
        (rv_access_dim,) = pattern.iter_ids_array[1]
        all_ops = layer_state.get_current_ops()
        cur_op = all_ops[op_id]
        inputs = layer_state[cur_op].op.input_tensors
        data = inputs[data_id.value]
        weight = inputs[weight_id.value]
        siv = layer_state[cur_op].axis()[siv_id.value]
        num_axis = len(layer_state[cur_op].axis())
        riv = layer_state[cur_op].reduce_axis()[riv_id.value]

        data_op_id = -1
        weight_op_id = -1
        for i, op in enumerate(all_ops):
            if op == data.op:
                data_op_id = i
            if op == weight.op:
                weight_op_id = i
        assert (data_op_id >= 0) and (weight_op_id >= 0)

        # parameter is GroupingParameter
        assert isinstance(parameter, GroupingParameter)
        explicit = False
        groups = int(parameter.groups)
        s_extent = int(siv.dom.extent)
        r_extent = int(riv.dom.extent)
        s_factor = (s_extent + groups - 1) // groups
        r_factor = (r_extent + groups - 1) // groups

        r1 = TransformRecord(
            op_id,
            [siv_id.value],
            "fold",
            json.dumps({"explicit": explicit, "factor": s_factor}),
        )
        r2 = TransformRecord(
            op_id,
            [riv_id.value + num_axis],
            "fold",
            json.dumps({"explicit": explicit, "factor": r_factor}),
        )
        r3 = TransformRecord(
            op_id,
            [riv_id.value + num_axis + 1],
            "eliminate",
            json.dumps({"explicit": explicit, "factor": s_factor}),
        )
        r4 = TransformRecord(
            weight_op_id,
            [rv_access_dim.value],
            "eliminate",
            json.dumps({"explicit": explicit, "factor": r_factor}),
        )

        # new_fsm_state = fsm_state.proceed_transform([r1, r2, r3, r4])
        # for r in [r1, r2, r3, r4]:
        #     self.applier.apply(layer_state, r)
        # return new_fsm_state
        return [r1, r2, r3, r4]


class GroupingParameterSpace(ParameterSpace):
    def __init__(self, layer_state, op_id, pattern):
        super(GroupingParameterSpace, self).__init__(layer_state, op_id, pattern)
        siv_id, riv_id = pattern.iter_ids_array[0]
        all_ops = layer_state.get_current_ops()
        cur_op = all_ops[op_id]
        siv = layer_state[cur_op].axis()[siv_id.value]
        riv = layer_state[cur_op].reduce_axis()[riv_id.value]

        s_extent = int(siv.dom.extent)
        r_extent = int(riv.dom.extent)
        groups = get_factor_lst(min(s_extent, r_extent))

        self.space = groups

    def random_get(self):
        choice = np.random.randint(0, len(self.space))
        return GroupingParameter(choice, self.space[choice])

    def rl_get(self):
        return super().rl_get()

    def feedback(self, parameter):
        return super().feedback(parameter)

    def update(self):
        return super().update()


class GroupingParameter(Parameter):
    def __init__(self, choice_id, groups):
        super(GroupingParameter, self).__init__(choice_id)
        self.groups = groups


class GroupingAction(Action):
    def __init__(self, layer_state, op_id, pattern):
        super(GroupingAction, self).__init__(
            "grouping",
            layer_state,
            op_id,
            pattern,
            GroupingParameterSpace(layer_state, op_id, pattern),
        )

    @staticmethod
    def applicable(layer_state, op_id):
        return GroupingGuarder.applicable(layer_state, op_id)

    def apply(self, parameter):
        return GroupingActor.apply(
            self.layer_state, self.op_id, self.pattern, parameter
        )
