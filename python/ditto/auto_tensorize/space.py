"""The space definition for tiling and fusion"""
from ..dse import (
    BaseItem,
    BaseCartSpace,
    SplitItem,
    SplitSpace,
    ChooseItem,
    ChooseSpace,
    PermuteItem,
    PermuteSpace,
)
from .iter_graph import IterGraph


class FusionTileSpace(BaseCartSpace):
    """Space for tiling for fusion."""

    def __init__(self, iter_graph, substantial=16):
        """
        Args:
            iter_graph (ditto.auto_schedule.hyper_fusion.IterGraph):
                The iterator graph.
            substantial (int): the minimal trip count of loops to be tiled
        """
        super(FusionTileSpace, self).__init__()
        assert isinstance(iter_graph, IterGraph)
        self.iter_graph = iter_graph
        # empty choices
        self.choices = []
        # tiling
        self.first_op_iters = iter_graph.getInitialFirstOpIters()
        self.second_op_iters = iter_graph.getInitialSecondOpIters()
        shared_iters = set()
        for (iter1, iter2) in iter_graph.getInitialShareIterPairs():
            shared_iters.add(iter1)
        for iv in self.first_op_iters:
            if (iv in shared_iters) or (iv.ext < substantial):
                if iv.isSpatial():
                    self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(
                        iv.ext, 2, mandatory_choices=[(iv.ext, 1)]
                    )
                elif iv.isReduce():
                    self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(
                        iv.ext, 2, mandatory_choices=[(1, iv.ext)]
                    )
            else:
                self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(iv.ext, 2)

        attach_pos_list = []
        for i, iv in enumerate(self.second_op_iters):
            if iv.ext < substantial:
                if iv.isSpatial():
                    self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(
                        iv.ext, 2, mandatory_choices=[(iv.ext, 1)]
                    )
                elif iv.isReduce():
                    self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(
                        iv.ext, 2, mandatory_choices=[(1, iv.ext)]
                    )
            else:
                attach_pos_list.append(i)
                self.subspaces[f"split-{iv}({hash(iv)})"] = SplitSpace(iv.ext, 2)

        # fusion
        self.subspaces["fuse"] = ChooseSpace(mandatory_choices=attach_pos_list)

        # reorder
        self.subspaces["reorder"] = PermuteSpace(
            num_elems=len(self.second_op_iters), hit_mask=attach_pos_list
        )  # why?

    @property
    def all_iter_graphs(self):
        ret = []
        for item in self.all_items:
            first_op_factors = []
            for iv in self.first_op_iters:
                split_item = item[f"split-{iv}({hash(iv)})"]
                first_op_factors.append(split_item[1])
            second_op_factors = []
            for iv in self.second_op_iters:
                split_item = item[f"split-{iv}({hash(iv)})"]
                second_op_factors.append(split_item[1])
            order = item["reorder"].order
            attach_pos = item["fuse"].item

            iter_graph = self.iter_graph.regenerate()
            iter_graph.setFirstOpTiling(first_op_factors)
            iter_graph.setSecondOpTiling(second_op_factors)
            iter_graph.permute(order)
            iter_graph.fuseLoops(attach_pos)
            ret.append(iter_graph)
        return ret
