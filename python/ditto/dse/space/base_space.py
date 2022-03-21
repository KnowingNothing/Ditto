"""The basic search space objects"""
from ...utils import ditto_logger
from ditto import utils


class BaseItem(object):
    """The base item for search space."""

    pass


class BaseSpace(object):
    """The base class for space object."""

    def __init__(self):
        self.name = ""
        self.choices = None

    @property
    def all_items(self):
        assert self.initialized(), f"The space {self.name} is not initialized."
        if len(self) > 100000:
            ditto_logger.warn(
                f"Attempt to retrive {len(self)} items from the design space!"
            )
        return self.choices

    def initialized(self):
        return isinstance(self.choices, (list, tuple))

    def __getitem__(self, items):
        return self.choices[items]

    def __iter__(self):
        return self.choices.__iter__()

    def __next__(self):
        return self.choices.__next__()

    def __len__(self):
        assert self.initialized(), f"The space {self.name} is not initialized."
        return self.choices.__len__()


class BaseCartSpace(object):
    """The base class for space with recursive subspaces."""

    def __init__(self):
        self.name = ""
        self.subspaces = {}

    @property
    def all_items(self):
        if len(self) > 100000:
            ditto_logger.warn(
                f"Attempt to retrive {len(self)} items from the design space!"
            )
        space_shape = []
        for key, subspace in self.subspaces.items():
            space_shape.append((key, len(subspace)))
        dim = len(space_shape)
        space_factors = [[k, 1] for (k, _) in space_shape]
        for i in range(0, dim - 1):
            space_factors[dim - i - 2][1] = (
                space_shape[dim - i - 1][1] * space_factors[dim - i - 1][1]
            )
        ids = range(len(self))

        def _inner(idx):
            item = {}
            v = idx
            for k, f in space_factors:
                outer = v // f
                inner = v % f
                item[k] = self.subspaces[k].all_items[outer]
                v = inner
            return item

        return utils.parallel_map(_inner, ids, True)

    def __getitem__(self, items):
        return self.subspaces[items]

    def __iter__(self):
        return self.subspaces.__iter__()

    def __next__(self):
        return self.subspaces.__next__()

    def __len__(self):
        ret = 1
        for k, v in self.subspaces.items():
            ret = ret * len(v)
        return ret
