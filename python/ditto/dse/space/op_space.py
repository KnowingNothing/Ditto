"""Search space for one operator."""
import itertools
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
        if mandatory_choices is not None:
            choices = []
            for it in mandatory_choices:
                if isinstance(it, ChooseItem):
                    choices.append(it)
                else:
                    choices.append(ChooseItem(it))
            self.choices = choices
        else:
            assert isinstance(lower, int)
            assert isinstance(upper, int)
            choices = list(range(lower, upper))
            self.choices = [ChooseItem(it) for it in choices]


class PermuteItem(BaseItem):
    """PermuteItem for loop reordering for operator
    """

    def __init__(self, order):
        """
        Args:
            order (List[int]): the index list
        """
        super(PermuteItem, self).__init__()
        self.order = order


class PermuteSpace(BaseSpace):
    """PermuteSpace for reordering loops for one operator.
    """

    def __init__(self, num_elems=None, hit_mask=None, mandatory_choices=None):
        """
        Args:
            num_elems (int): how many elements in total
            hit_mask (List[int]): if i in hit_mask, it means the i-th value
                can be reordered. Otherwise, it can't be reordered.
            mandatory_choices (List[PermuteItem]): the choices
                given by the user.
        """
        if mandatory_choices is not None:
            choices = []
            for it in mandatory_choices:
                if isinstance(it, PermuteItem):
                    choices.append(it)
                else:
                    choices.append(PermuteItem(it))
            self.choices = choices
        else:
            choices = []
            assert isinstance(num_elems, int)
            assert max(hit_mask) < num_elems and min(hit_mask) >= 0
            hit_mask = list(sorted(hit_mask))
            for p in itertools.permutations(hit_mask):
                it = list(range(num_elems))
                for idx, v in zip(hit_mask, p):
                    it[idx] = v
                choices.append(it)
            self.choices = [ChooseItem(it) for it in choices]
