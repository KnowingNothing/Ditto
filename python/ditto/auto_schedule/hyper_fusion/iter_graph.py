"""IterGraph Implementation"""
from .utils import ceil


class IterVar(object):
    """The internal IterVar object of hyper fusion.
        The name of IterVar is used as hash key.
        The IterVar is assumed to be normalized to [0, ext).
    """

    def __init__(self, name, ext=None):
        self.name = name
        self.ext = ext

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return f"IterVar({self.name}, org_ext={self.ext})"

    def __repr__(self):
        return self.__str__()


class Relation(object):
    """The base relationship object for IterVar.
    """
    pass


class SplitRelation(Relation):
    """Split relationship
    """

    def __init__(self, parent, left, right):
        self.parent = parent
        self.left = left
        self.right = right


class ShareRelation(Relation):
    """Share relationship
    """

    def __init__(self, iter1, iter2):
        self.iter1 = iter1
        self.iter2 = iter2


class AttachRelation(Relation):
    """Attach relationship
    """

    def __init__(self, head, remain, original):
        """Head is from the second op.
            Remain is the remaining inner private iter
            of the first op after fusion.
            Original is the original parent iter
            of the first op before fusion.
        """
        self.head = head
        self.remain = remain
        self.original = original


class IterGraph(object):
    """The iterator graph object.
    """

    def __init__(self, first_op_iters, second_op_iters, share_iter_pairs):
        self.common_iters = []
        self.first_op_private_iters = []
        self.first_op_num_loops = 0
        self.second_op_private_iters = []
        self.second_op_num_loops = 0
        # i = i1*Ti + i2
        # denote two iters are split from one
        self.split_iter_relations = []  # List[SplitRelation]
        # denote two iters access the same tensor
        # along the same dimension
        self.shared_iter_relations = []  # List[ShareRelation]
        # (i j) is a share relation
        # (i i1 i2 Ti) and (j j1 j2 Tj) are split relations
        # then j1 and i2, i1 and j2 are attach relations
        # denoted as (i j1 i2) and (j i1 j2)
        self.attach_iter_relations = []  # List[AttachRelation]
        self._build(first_op_iters, second_op_iters, share_iter_pairs)

    def _build(self, first_op_iters, second_op_iters, share_iter_pairs):
        """Build the iter graph using
            the iterators of the first cubic op
            and the second cubic op.

        Args:
            first_op_iters (List[IterVar]): iterators of the first cubic op
            second_op_iters (List[IterVar]): iterators of the second cubic op
            share_iter_pairs (List[Tuple(IterVar, IterVar)]): (first op iter, second op iter)
        """
        self.first_op_private_iters = first_op_iters
        self.second_op_private_iters = second_op_iters
        self.first_op_num_loops = len(first_op_iters)
        self.second_op_num_loops = len(second_op_iters)
        for (first, second) in share_iter_pairs:
            self.shared_iter_relations.append(ShareRelation(first, second))

    def setFirstOpTiling(self, first_op_tile_factors):
        """Set the tiling factors for the first op.

        Args:
            first_op_tile_factors (List[int]): the tiling factors
                                                for the first op.
        """
        assert len(first_op_tile_factors) == len(self.first_op_private_iters)
        outer_iters = [IterVar(f"{iv.name}.outer")
                       for iv in self.first_op_private_iters]
        inner_iters = [IterVar(f"{iv.name}.inner", factor) for iv, factor in zip(
            self.first_op_private_iters, first_op_tile_factors)]
        for outer, inner, iv in zip(outer_iters, inner_iters, self.first_op_private_iters):
            self.split_iter_relations.append(SplitRelation(iv, outer, inner))
            for i in range(len(self.shared_iter_relations)):
                share_rel = self.shared_iter_relations[i]
                if share_rel.iter1 == iv:
                    self.shared_iter_relations[i].iter1 = outer

        self.first_op_private_iters = outer_iters + inner_iters

    def setSecondOpTiling(self, second_op_tile_factors):
        """Set the tiling factors for the second op.

        Args:
            second_op_tile_factors (List[int]): the tiling factors
                                                for the second op.
        """
        assert len(second_op_tile_factors) == len(self.second_op_private_iters)
        outer_iters = [IterVar(f"{iv.name}.outer")
                       for iv in self.second_op_private_iters]
        inner_iters = [IterVar(f"{iv.name}.inner", factor) for iv, factor in zip(
            self.second_op_private_iters, second_op_tile_factors)]
        for outer, inner, iv in zip(outer_iters, inner_iters, self.second_op_private_iters):
            self.split_iter_relations.append(SplitRelation(iv, outer, inner))
            for i in range(len(self.shared_iter_relations)):
                share_rel = self.shared_iter_relations[i]
                if share_rel.iter2 == iv:
                    self.shared_iter_relations[i].iter2 = outer

        self.second_op_private_iters = outer_iters + inner_iters

    def permute(self, order):
        """Change the loop order of the second op.

        Args:
            order (List[int]): new order of the iterators
        """
        assert len(order) == len(self.second_op_private_iters)
        sorted_values = list(sorted(order))
        assert tuple(sorted_values) == tuple(
            range(len(order))), "some iterator is missing"
        permuted_loops = [self.second_op_private_iters[i] for i in order]
        self.second_op_private_iters = permuted_loops

    def fuseLoops(self, attach_pos):
        """Fuse the two loop nests.

        Args:
            attach_pos (int): the index of the loop of the second op
                              to fuse at.
        """
        assert attach_pos < self.second_op_num_loops, "fusion out of bound"
        common_iters = self.second_op_private_iters[:attach_pos + 1]
        second_op_private_iters = self.second_op_private_iters[attach_pos + 1:]

        lookup_set = set(common_iters)
        first_op_private_iters = []
        for iv in self.first_op_private_iters:
            eliminate = False
            for i in range(len(self.shared_iter_relations)):
                share_rel = self.shared_iter_relations[i]
                if share_rel.iter1 == iv and share_rel.iter2 in lookup_set:
                    # find the split inner one
                    for j in range(len(self.split_iter_relations)):
                        split_rel = self.split_iter_relations[j]
                        if split_rel.left == iv:
                            remain = IterVar(split_rel.right.name, None)
                            self.attach_iter_relations.append(AttachRelation(
                                share_rel.iter2, remain, split_rel.parent))
                    eliminate = True
            if not eliminate:
                first_op_private_iters.append(iv)

        self.common_iters = common_iters
        self.first_op_private_iters = first_op_private_iters
        self.second_op_private_iters = second_op_private_iters

    def inferBound(self):
        """Get the bounds of all existing iterators."""
        bounds = {}
        for iter in self.common_iters + self.first_op_private_iters + self.second_op_private_iters:
            if iter.ext is not None:
                bounds[iter] = iter.ext
        for split_rel in self.split_iter_relations:
            assert split_rel.parent.ext is not None, f"The root iter_var {split_rel.parent}'s bound is unknown."
            bounds[split_rel.parent] = split_rel.parent.ext
            if split_rel.left.ext is None:
                assert split_rel.left not in bounds
                assert split_rel.right.ext is not None
                bounds[split_rel.left] = ceil(
                    split_rel.parent.ext, split_rel.right.ext)
        for attach_rel in self.attach_iter_relations:
            head = attach_rel.head
            remain = attach_rel.remain
            original = attach_rel.original
            assert original.ext is not None
            assert head in bounds, f"bounds for {head} is not inferred successfully."
            bounds[remain] = ceil(original.ext, bounds[head])
        return bounds

    def commonLoops(self):
        return self.common_iters

    def firstLoops(self):
        return self.first_op_private_iters

    def secondLoops(self):
        return self.second_op_private_iters
