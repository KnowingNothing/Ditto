from collections import namedtuple, deque


class State(object):
    def __init__(self):
        self.transition_history = []


Transition = namedtuple("Transition", ["action"])


StateTransition = namedtuple("StateTransition", ["state", "transition"])


class StateMachnie(object):
    def __init__(self, init_states):
        self.init_states = init_states
        self.state_transition_que = deque()
        self.final_states = []