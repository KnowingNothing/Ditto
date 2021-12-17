"""The space definition for tiling and fusion"""
from ...dse import (
    BaseSpace, SplitItem, SplitSpace, ChooseItem, ChooseSpace)
from .iter_graph import IterGraph


class FusionTileSpace(BaseSpace):
    """Space for tiling for fusion."""
    def __init__(self, iter_graph):
        """
        Args:
            iter_graph (ditto.auto_schedule.hyper_fusion.IterGraph):
                The iterator graph.
        """
        super(FusionTileSpace, self).__init__()
        assert isinstance(iter_graph, IterGraph)
        self.choices = []
        self.first_op_iters = iter_graph.get_initial_first_op_iters()
        self.second_op_iters = iter_graph.get_initial_second_op_iters()
        for iv in self.first_op_iters + self.second_op_iters:
            self.subspaces[iv] = SplitSpace(iv.ext, 2)
        