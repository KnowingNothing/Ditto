from collections import namedtuple, deque

from .state import create_layer_state
from .action.record import TransformApplier
from .action import GroupingAction, NopAction


class State(object):
    def __init__(self, layer):
        self.layer = layer
        self.action_history = []
        self.transform_history = []

    def proceed_action(self, record_list):
        new_state = State(self.layer)
        new_state.action_history.extend(self.action_history)
        new_state.action_history.extend(record_list)
        new_state.transform_history.extend(self.transform_history)
        return new_state

    def proceed_transform(self, record_list):
        new_state = State(self.layer)
        new_state.action_history.extend(self.action_history)
        new_state.transform_history.extend(self.transform_history)
        new_state.transform_history.extend(record_list)
        return new_state

    def replay(self):
        layer_state = create_layer_state(self.layer)
        applier = TransformApplier()
        for r in self.transform_history:
            applier.apply(layer_state, r)
        return layer_state


class StateMachnie(object):
    def __init__(self, init_states):
        self.init_states = init_states
        self.action_classes = [
            GroupingAction,
            NopAction
        ]
        self.allow_combines = {
            GroupingAction: [],
            NopAction: []
        }
        self.op_id_shift = {
            GroupingAction: 1,
            NopAction: 1
        }

    def run_op(self, state, op_id):
        new_states = []

        def _traverse(action_class, cur_state):
            layer_state = cur_state.replay()
            applicable, patterns = action_class.applicable(layer_state, op_id)
            if (applicable):
                for pattern in patterns:
                    action = action_class(layer_state, op_id, pattern)
                    # random parameter just to proceed the generation
                    parameter = action.random_param()
                    new_state = cur_state.proceed_action([action.to_record()])
                    transform_records = action.apply(parameter)
                    new_state = new_state.proceed_transform(transform_records)
                    for next_action_class in self.allow_combines[action_class]:
                        _traverse(next_action_class, new_state)
                    next_op_id = op_id - self.op_id_shift[action_class]
                    new_states.append((new_state, next_op_id))

        for action_class in self.action_classes:
            _traverse(action_class, state)
        return new_states

    def run(self):
        reachable_states = []
        trans_que = deque()
        for state in self.init_states:
            layer_state = state.replay()
            all_ops = layer_state.get_current_ops()
            num_ops = len(all_ops)
            trans_que.append((state, num_ops - 1))
        while len(trans_que):
            (cur_state, op_id) = trans_que.popleft()
            if op_id < 0:
                reachable_states.append(cur_state)
            else:
                new_options = self.run_op(cur_state, op_id)
                trans_que.extend(new_options)

        return reachable_states
