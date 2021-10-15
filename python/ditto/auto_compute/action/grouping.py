from typing_extensions import ParamSpec
from .action_base import Guarder, Actor, Action, ParameterSpace, Parameter


class GroupingGuarder(Guarder):
    pass


class GroupingActor(Actor):
    pass


class GroupingParameterSpace(ParameterSpace):
    pass


class GroupingParameter(Parameter):
    pass


class GroupingAction(Action):
    def __init__(self):
        super(GroupingAction, self).__init__()
        self.guarder = GroupingGuarder()
        self.actor = GroupingActor()
        self.para_space = GroupingParameterSpace()
        