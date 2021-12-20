"""The basic search space objects"""


class BaseItem(object):
    """The base item for search space."""
    pass


class BaseSpace(object):
    """The base class for space object."""

    def __init__(self):
        self.name = ""
        self.choices = None

    def all_items(self):
        assert self.initialized(), f"The space {self.name} is not initialized."
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
    
    def all_items(self):
        keys = list(self.subspaces.keys())
        num_space = len(keys)
        ret = []
        def _inner(idx, cur):
            if idx == num_space:
                ret.append(cur)
                return
            else:
                for item in self.subspaces[keys[idx]].all_items():
                    next_item = {}
                    next_item.update(cur)
                    next_item[keys[idx]] = item
                    _inner(idx + 1, next_item)
        _inner(0, {})
        return ret      

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
