"""Search space for one iterator."""
from .base_space import BaseItem, BaseSpace
from .utils import any_factor_split


class SplitItem(BaseItem):
    """SplitItem for tiling one iterator"""

    def __init__(self, factors):
        """
        Args:
            factors (List[int]): tiling factors
        """
        super(SplitItem, self).__init__()
        assert isinstance(factors, (list, tuple))
        self.factors = factors

    def __getitem__(self, items):
        return self.factors[items]

    def __iter__(self):
        return self.factors.__iter__()

    def __next__(self):
        return self.factors.__next__()

    def __len__(self):
        return self.factors.__len__()


class SplitSpace(BaseSpace):
    """SplitSpace for tiling one iterator."""

    def __init__(self, extent=None, parts=None, mandatory_choices=None):
        """
        Args:
            extent (int): the extent of one iterator
            parts (int): how many parts to split
            mandatory_choices (List[SplitItem]): the tile choices
                given by the user.
        """
        if mandatory_choices is not None:
            choices = []
            for it in mandatory_choices:
                assert len(it) == parts, f"Expect {parts} parts, but get {len(it)}."
                if isinstance(it, SplitItem):
                    choices.append(it)
                else:
                    choices.append(SplitItem(it))
            self.choices = choices
        else:
            assert isinstance(extent, int)
            assert isinstance(parts, int)
            choices = any_factor_split(extent, parts)
            self.choices = [SplitItem(it) for it in choices]
