


class Guarder(object):
    def applicable(self, layer_state, cur_op, fsm_state):
        raise NotImplementedError()


class Actor(object):
    def apply(self, layer_state, cur_op, fsm_state):
        raise NotImplementedError()


class ParameterSpace(object):
    def initialize(self, layer_state, cur_op, fsm_state):
        raise NotImplementedError()
    
    def random_get(self, layer_state, cur_op, fsm_state):
        raise NotImplementedError()
    
    def rl_get(self, layer_state, cur_op, fsm_state):
        raise NotImplementedError()
    
    def feedback(self, layer_state, cur_op, fsm_state, parameter, reward):
        raise NotImplementedError()
    
    def update(self):
        raise NotImplementedError()


class Parameter(object):
    pass    


class Action(object):
    pass
