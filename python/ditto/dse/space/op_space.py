"""Search space for one operator."""
from .base_space import BaseItem, BaseSpace


class ChooseItem(BaseItem):
    """ChooseItem for choose one item for operator
    """

    def __init__(self, item):
        """
        Args:
            item (any): the chosen item
        """
        super(ChooseItem, self).__init__()
        self.item = item


class ChooseSpace(BaseSpace):
    """ChooseSpace for choose one item for one operator.
        By default, choose a value from [lower, upper).
    """

    def __init__(self, lower=None, upper=None, mandatory_choices=None):
        """
        Args:
            lower (int): the lower bound of choice range
            upper (int): the upper bound of choice range
            mandatory_choices (List[ChooseItem]): the choices
                given by the user.
        """
        assert isinstance(lower, int)
        assert isinstance(upper, int)
        if mandatory_choices is not None:
            choices = []
            for it in mandatory_choices:
                if isinstance(it, ChooseItem):
                    choices.append(it)
                else:
                    choices.append(ChooseItem(it))
            self.choices = choices
        else:
            choices = list(range(lower, upper))
            self.choices = [ChooseItem(it) for it in choices]
