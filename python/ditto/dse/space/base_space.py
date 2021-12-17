"""The basic search space objects"""


class BaseItem(object):
    """The base item for search space."""
    pass


class BaseSpace(object):
    """The base class for space object."""

    def __init__(self):
        self.choices = None
        self.subspaces = {}
        
    def initialized(self):
        return isinstance(self.choices, (list, tuple))
    
    def __getitem__(self, items):
        return self.choices[items]
    
    def __iter__(self):
        return self.choices.__iter__()
    
    def __next__(self):
        return self.choices.__next__()
    
    def __len__(self):
        return self.choices.__len__()
    

