import json
from .pattern import Pattern
from .record import ActionRecord


class Guarder(object):
    @staticmethod
    def applicable(layer_state, op_id):
        raise NotImplementedError()


class Actor(object):
    @staticmethod
    def apply(layer_state, op_id, pattern, parameter):
        raise NotImplementedError()


class ParameterSpace(object):
    def __init__(self, layer_state, op_id, pattern):
        self.layer_state = layer_state
        self.op_id = op_id
        self.pattern = pattern

    def random_get(self):
        raise NotImplementedError()

    def rl_get(self):
        raise NotImplementedError()

    def feedback(self, parameter):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()


class Parameter(object):
    def __init__(self, choice_id):
        self.choice_id = choice_id


class Action(object):
    def __init__(self, action_id, layer_state, op_id, pattern, param_space):
        self.action_id = action_id
        self.layer_state = layer_state
        self.op_id = op_id
        self.pattern = pattern
        self.param_space = param_space

    @staticmethod
    def applicable(layer_state, op_id):
        raise NotImplementedError()

    def random_param(self):
        return self.param_space.random_get()

    def to_record(self):
        return ActionRecord(
            self.op_id, self.action_id, json.dumps(self.pattern.to_json()))

    def apply(self, parameter):
        raise NotImplementedError()


class NopParameterSpace(ParameterSpace):
    def __init__(self, layer_state, op_id, pattern):
        super(NopParameterSpace, self).__init__(
            layer_state, op_id, pattern)
        self.space = []

    def random_get(self):
        return NopParameter(-1)

    def rl_get(self):
        return NopParameter(-1)

    def feedback(self, parameter):
        pass

    def update(self):
        pass


class NopParameter(Parameter):
    def __init__(self, choice_id):
        super(NopParameter, self).__init__(choice_id)


class NopAction(Action):
    def __init__(self, layer_state, op_id, pattern):
        super(NopAction, self).__init__("nop", layer_state, op_id,
                                        pattern, NopParameterSpace(layer_state, op_id, pattern))

    @staticmethod
    def applicable(layer_state, op_id):
        return True, [Pattern.empty_pattern()]

    def apply(self, parameter):
        return []
