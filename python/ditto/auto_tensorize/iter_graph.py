"""IterGraph Implementation"""
import tvm
from .utils import ceil
from typing import *
from ditto import utils


IV_TYPE_SPATIAL = "IV_TYPE_SPATIAL"
IV_TYPE_REDUCE = "IV_TYPE_REDUCE"


class IterVar(object):
    """The internal IterVar object of hyper fusion.
    The name of IterVar is used as hash key.
    The IterVar is assumed to be normalized to [0, ext).
    """

    def __init__(self, name, idx, ext=None, iv_type=None):
        self.name = name
        self.idx = idx
        self.ext = ext
        self.iv_type = iv_type

    def isSpatial(self):
        return self.iv_type == IV_TYPE_SPATIAL

    def isReduce(self):
        return self.iv_type == IV_TYPE_REDUCE

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return f"IterVar({self.name}, org_ext={self.ext})"

    def __repr__(self):
        return self.__str__()


class Relation(object):
    """The base relationship object for IterVar."""

    pass


class SplitRelation(Relation):
    """Split relationship"""

    def __init__(self, parent, left, right):
        self.parent = parent
        self.left = left
        self.right = right


class ShareRelation(Relation):
    """Share relationship"""

    def __init__(self, iter1, iter2):
        self.iter1 = iter1
        self.iter2 = iter2


class AttachRelation(Relation):
    """Attach relationship"""

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


class AccessFunc(object):
    """The object to model an access function."""

    def __init__(
        self,
        indices_list: List[List[tvm.ir.PrimExpr]],
        iters_mapping: Dict[tvm.tir.Var, IterVar],
    ):
        self.indices_list = indices_list
        self.iters_mapping = iters_mapping

    def inferDataSize(self, bounds):
        ret = []
        range_map = {}
        for k, v in self.iters_mapping.items():
            range_map[k] = tvm.ir.Range(0, bounds[v])
        for indices in self.indices_list:
            ranges = utils.infer_range(indices, range_map)
            shape = [r.min.value + r.extent.value for r in ranges]  # why?
            size = utils.product(shape)
            ret.append(size)
        return ret


class IterGraph(object):
    """The iterator graph object."""

    def __init__(
        self,
        first_op_iters: Sequence[IterVar],
        second_op_iters: Sequence[IterVar],
        share_iter_pairs: Sequence[Tuple[IterVar, IterVar]],
        first_op_read_access_funcs: Sequence[AccessFunc],
        second_op_read_access_funcs: Sequence[AccessFunc],
        first_write_access_func: AccessFunc,
        second_write_access_func: AccessFunc,
        read_producer_pos: int,
    ):
        # these are not mutable
        # we produce new values for mutable states
        # so don't worry that these values will be changed
        self._initial_first_op_iters = first_op_iters
        self._initial_second_op_iters = second_op_iters
        self._initial_share_iter_pairs = share_iter_pairs
        self._initial_first_op_access_funcs = first_op_read_access_funcs
        self._initial_second_op_access_funcs = second_op_read_access_funcs
        self._initial_first_write_access_func = first_write_access_func
        self._initial_second_write_access_func = second_write_access_func
        self._initial_read_producer_pos = read_producer_pos
        # the following are mutable states
        self.common_iters = []
        self.first_op_private_iters = []
        self.first_op_num_loops = 0
        self.second_op_private_iters = []
        self.second_op_num_loops = 0
        self.first_op_read_access_funcs = []
        self.second_op_read_access_funcs = []
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
        self._build(
            first_op_iters,
            second_op_iters,
            share_iter_pairs,
            first_op_read_access_funcs,
            second_op_read_access_funcs,
            first_write_access_func,
            second_write_access_func,
            read_producer_pos,
        )

    def _build(
        self,
        first_op_iters: Sequence[IterVar],
        second_op_iters: Sequence[IterVar],
        share_iter_pairs: Sequence[Tuple[IterVar, IterVar]],
        first_op_read_access_funcs: Sequence[AccessFunc],
        second_op_read_access_funcs: Sequence[AccessFunc],
        first_write_access_func: AccessFunc,
        second_write_access_func: AccessFunc,
        read_producer_pos: int,
    ):
        """Build the iter graph using
            the iterators of the first cubic op
            and the second cubic op.

        Args:
            first_op_iters (Sequence[IterVar]): iters of the first op
            second_op_iters (Sequence[IterVar]): iters of the second op
            share_iter_pairs (Sequence[Tuple[IterVar, IterVar]]): shared iters tuples
            first_op_read_access_funcs (Sequence[AccessFunc]): access funcs of the first op
            second_op_read_access_funcs (Sequence[AccessFunc]): access funcs of the second op
            first_write_access_func (AccessFunc)
            second_write_access_func (AccessFunc)
            read_producer_pos (int)
        """
        self.first_op_private_iters = first_op_iters
        self.second_op_private_iters = second_op_iters
        self.first_op_num_loops = len(first_op_iters)
        self.second_op_num_loops = len(second_op_iters)
        self.first_op_read_access_funcs = first_op_read_access_funcs
        self.second_op_read_access_funcs = second_op_read_access_funcs
        for (first, second) in share_iter_pairs:
            self.shared_iter_relations.append(ShareRelation(first, second))
        self.first_op_write_access_func = first_write_access_func
        self.second_op_write_access_func = second_write_access_func
        self.read_producer_pos = read_producer_pos

    def regenerate(self):
        """Get another IterGraph that is the same as the initial one."""
        return IterGraph(
            self._initial_first_op_iters,
            self._initial_second_op_iters,
            self._initial_share_iter_pairs,
            self._initial_first_op_access_funcs,
            self._initial_second_op_access_funcs,
            self._initial_first_write_access_func,
            self._initial_second_write_access_func,
            self._initial_read_producer_pos,
        )

    def getInitialFirstOpIters(self):
        return [x for x in self._initial_first_op_iters]

    def getInitialSecondOpIters(self):
        return [x for x in self._initial_second_op_iters]

    def getInitialShareIterPairs(self):
        return [x for x in self._initial_share_iter_pairs]

    def getInitialFirstOpReadAccessFuncs(self):
        return [x for x in self._initial_first_op_access_funcs]

    def getInitialSecondOpReadAccessFuncs(self):
        return [x for x in self._initial_second_op_access_funcs]

    def setFirstOpTiling(self, first_op_tile_factors: Sequence[int]):
        """Set the tiling factors for the first op.

        Args:
            first_op_tile_factors (List[int]): the tiling factors
                                                for the first op.
        """
        assert len(first_op_tile_factors) == len(self.first_op_private_iters)
        outer_iters = [
            IterVar(f"{iv.name}.outer", iv.idx, ext=None, iv_type=iv.iv_type)
            for iv in self.first_op_private_iters
        ]
        inner_iters = [
            IterVar(f"{iv.name}.inner", iv.idx, ext=factor, iv_type=iv.iv_type)
            for iv, factor in zip(self.first_op_private_iters, first_op_tile_factors)
        ]
        # map from original iter to inner iter
        convert_to_inner = {}
        for org, inner in zip(self.first_op_private_iters, inner_iters):
            convert_to_inner[org] = inner
        for outer, inner, iv in zip(
            outer_iters, inner_iters, self.first_op_private_iters
        ):
            self.split_iter_relations.append(SplitRelation(iv, outer, inner))
            for i in range(len(self.shared_iter_relations)):
                share_rel = self.shared_iter_relations[i]
                if share_rel.iter1 == iv:
                    self.shared_iter_relations[i].iter1 = outer

        self.first_op_private_iters = outer_iters + inner_iters

    def setSecondOpTiling(self, second_op_tile_factors: Sequence[int]):
        """Set the tiling factors for the second op.

        Args:
            second_op_tile_factors (List[int]): the tiling factors
                                                for the second op.
        """
        assert len(second_op_tile_factors) == len(self.second_op_private_iters)
        outer_iters = [
            IterVar(f"{iv.name}.outer", iv.idx, ext=None, iv_type=iv.iv_type)
            for iv in self.second_op_private_iters
        ]
        inner_iters = [
            IterVar(f"{iv.name}.inner", iv.idx, ext=factor, iv_type=iv.iv_type)
            for iv, factor in zip(self.second_op_private_iters, second_op_tile_factors)
        ]
        # map from original iter to inner iter
        convert_to_inner = {}
        for org, inner in zip(self.second_op_private_iters, inner_iters):
            convert_to_inner[org] = inner
        for outer, inner, iv in zip(
            outer_iters, inner_iters, self.second_op_private_iters
        ):
            self.split_iter_relations.append(SplitRelation(iv, outer, inner))
            for i in range(len(self.shared_iter_relations)):
                share_rel = self.shared_iter_relations[i]
                if share_rel.iter2 == iv:
                    self.shared_iter_relations[i].iter2 = outer

        self.second_op_private_iters = outer_iters + inner_iters

    def permute(self, order: Sequence[int]):
        """Change the loop order of the second op.

        Args:
            order (List[int]): new order of the iterators
        """
        length = len(self._initial_second_op_iters)
        assert (
            len(order) == length
        ), f"order={order}, iters={self._initial_second_op_iters}"
        sorted_values = list(sorted(order))
        assert tuple(sorted_values) == tuple(range(length)), "some iterator is missing"
        permuted_loops = [
            self.second_op_private_iters[i] for i in order
        ] + self.second_op_private_iters[length:]
        self.second_op_private_iters = permuted_loops

    def fuseLoops(self, attach_pos: int):
        """Fuse the two loop nests.

        Args:
            attach_pos (int): the index of the loop of the second op
                              to fuse at.
        """
        assert attach_pos < self.second_op_num_loops, "fusion out of bound"
        common_iters = self.second_op_private_iters[: attach_pos + 1]
        second_op_private_iters = self.second_op_private_iters[attach_pos + 1 :]

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
                            remain = IterVar(
                                split_rel.right.name,
                                split_rel.right.idx,
                                ext=None,
                                iv_type=split_rel.right.iv_type,
                            )
                            self.attach_iter_relations.append(
                                AttachRelation(
                                    share_rel.iter2, remain, split_rel.parent
                                )
                            )
                    eliminate = True
            if not eliminate:
                first_op_private_iters.append(iv)

        self.common_iters = common_iters
        self.first_op_private_iters = first_op_private_iters
        self.second_op_private_iters = second_op_private_iters

        first_op_iters_map = {x.name: x for x in first_op_private_iters}
        second_op_iters_map = {x.name: x for x in second_op_private_iters}

        new_first_op_access_funcs = []
        for func in self.first_op_read_access_funcs:
            new_mapping = {}
            # var 2 iterVar
            for k, v in func.iters_mapping.items():
                if (
                    f"{v.name}.outer" in first_op_iters_map
                    and f"{v.name}.inner" in first_op_iters_map
                ):
                    new_mapping[k] = first_op_iters_map[f"{v.name}.inner"]
                elif f"{v.name}.outer" in first_op_iters_map:
                    new_mapping[k] = first_op_iters_map[f"{v.name}.outer"]
                elif f"{v.name}.inner" in first_op_iters_map:
                    new_mapping[k] = first_op_iters_map[f"{v.name}.inner"]
                else:
                    raise ValueError(f"Can't find iterator: {v}.")
            new_first_op_access_funcs.append(AccessFunc(func.indices_list, new_mapping))
        self.first_op_read_access_funcs = new_first_op_access_funcs

        new_write_mapping = {}
        for k, v in self.first_op_write_access_func.iters_mapping.items():
            if (
                f"{v.name}.outer" in first_op_iters_map
                and f"{v.name}.inner" in first_op_iters_map
            ):
                new_write_mapping[k] = v
            elif f"{v.name}.outer" in first_op_iters_map:
                new_write_mapping[k] = first_op_iters_map[f"{v.name}.outer"]
            elif f"{v.name}.inner" in first_op_iters_map:
                new_write_mapping[k] = first_op_iters_map[f"{v.name}.inner"]
            else:
                raise ValueError(f"Can't find iterator: {v}.")
        self.first_op_write_access_func = AccessFunc(
            self.first_op_write_access_func.indices_list, new_write_mapping
        )

        new_second_op_access_funcs = []
        for func in self.second_op_read_access_funcs:
            new_mapping = {}
            for k, v in func.iters_mapping.items():
                if (
                    f"{v.name}.outer" in second_op_iters_map
                    and f"{v.name}.inner" in second_op_iters_map
                ):
                    new_mapping[k] = second_op_iters_map[f"{v.name}.inner"]
                elif f"{v.name}.outer" in second_op_iters_map:
                    new_mapping[k] = second_op_iters_map[f"{v.name}.outer"]
                elif f"{v.name}.inner" in second_op_iters_map:
                    new_mapping[k] = second_op_iters_map[f"{v.name}.inner"]
                else:
                    raise ValueError(f"Can't find iterator: {v}.")
            new_second_op_access_funcs.append(
                AccessFunc(func.indices_list, new_mapping)
            )
        self.second_op_read_access_funcs = new_second_op_access_funcs

        new_write_mapping = {}
        for k, v in self.second_op_write_access_func.iters_mapping.items():
            if (
                f"{v.name}.outer" in second_op_iters_map
                and f"{v.name}.inner" in second_op_iters_map
            ):
                new_write_mapping[k] = v
            elif f"{v.name}.outer" in second_op_iters_map:
                new_write_mapping[k] = second_op_iters_map[f"{v.name}.outer"]
            elif f"{v.name}.inner" in second_op_iters_map:
                new_write_mapping[k] = second_op_iters_map[f"{v.name}.inner"]
            else:
                raise ValueError(f"Can't find iterator: {v}.")
        self.second_op_write_access_func = AccessFunc(
            self.second_op_write_access_func.indices_list, new_write_mapping
        )

    def inferBound(self):
        """Get the bounds of all existing iterators."""
        bounds = {}
        for iter in (
            self.common_iters
            + self.first_op_private_iters
            + self.second_op_private_iters
        ):
            if iter.ext is not None:
                bounds[iter] = iter.ext
        for split_rel in self.split_iter_relations:
            assert (
                split_rel.parent.ext is not None
            ), f"The root iter_var {split_rel.parent}'s bound is unknown."
            bounds[split_rel.parent] = split_rel.parent.ext
            if split_rel.left.ext is None:
                assert split_rel.left not in bounds
                assert split_rel.right.ext is not None
                bounds[split_rel.left] = ceil(split_rel.parent.ext, split_rel.right.ext)
        for attach_rel in self.attach_iter_relations:
            head = attach_rel.head
            remain = attach_rel.remain
            original = attach_rel.original
            assert original.ext is not None
            assert head in bounds, f"bounds for {head} is not inferred successfully."
            bounds[remain] = ceil(original.ext, bounds[head])
        return bounds

    def getFirstOpReadAccessDataSize(self, bounds):
        ret = []
        for access in self.first_op_read_access_funcs:
            size_list = access.inferDataSize(bounds)
            ret.append(size_list)
        return ret

    def getFirstOpWriteAccessDataSize(self, bounds):
        return self.first_op_write_access_func.inferDataSize(bounds)

    def getSecondOpReadAccessDataSize(self, bounds):
        ret = []
        for access in self.second_op_read_access_funcs:
            size_list = access.inferDataSize(bounds)
            ret.append(size_list)
        return ret

    def getSecondOpWriteAccessDataSize(self, bounds):
        return self.second_op_write_access_func.inferDataSize(bounds)

    def getFirstOpSecondOpRelateInputPos(self):
        return self.read_producer_pos

    def commonLoops(self):
        return self.common_iters

    def firstLoops(self):
        return self.first_op_private_iters

    def secondLoops(self):
        return self.second_op_private_iters

    def redundantCommonLoops(self):
        ret = []
        common_loops = self.commonLoops()
        mark = {x.name: 0 for x in common_loops}
        for share_rel in self.shared_iter_relations:
            if f"{share_rel.iter2.name}.outer" in mark:
                mark[f"{share_rel.iter2.name}.outer"] += 1
        for l in common_loops:
            if mark[l.name] == 0:
                ret.append(l)
        return ret
