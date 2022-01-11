from typing import OrderedDict, Union, Optional
from dataclasses import dataclass

import numpy as np


# immutable
@dataclass(repr=True, frozen=True)
class Range:
    min: int
    extent: int

    @property
    def max(self):
        return self.min + self.extent - 1

    def copy(self):
        return Range(self.min, self.extent)


# immutable (except Iter.stage)
class Iter:
    def __init__(self, name: str, range: Union[Range, tuple], reduce=False):
        self._name = name
        self._range = range if isinstance(range, Range) else Range(*range)
        self._reduce = reduce

    @property
    def name(self):
        return self._name

    @property
    def range(self):
        return self._range

    @property
    def reduce(self):
        return self._reduce

    @property
    def iter_kind(self):
        return "reduce" if self.reduce else "spatial"

    def __repr__(self):
        return self.name


class IndexExpr:
    """
    affine index expressions
    e.g. i*3+j*4+2, where i and j are Iterators
    -> coeff_map: { i:3, j:4 }
    -> const_term: 2
    """
    def __init__(self, coeff_map: OrderedDict, const_term: int = 0):
        super().__init__()
        self.coeff_map: OrderedDict = coeff_map
        self.const_term = const_term

    @property
    def iters(self):
        return self.coeff_map.keys()

    @property
    def spatial_iters(self):
        return (it for it in self.iters if it.iter_kind == "spatial")

    def __repr__(self):
        terms = list()
        
        for it, coeff in self.coeff_map.items():
            if coeff == 1: terms.append(str(it))
            else: terms.append(f'{coeff}*{it}')
        
        if self.const_term != 0 or len(terms) == 0:
            terms.append(str(self.const_term))
        
        return " + ".join(terms)

    def copy(self):
        # iterators are immutable, not copied
        new_coeff_map = self.coeff_map.copy()
        return IndexExpr(new_coeff_map, self.const_term)

    # TODO: move into Step classes
    """ index transforms """

    # delete an iterator
    def delete_iter(self, iter: Iter):
        new_expr = self.copy()
        if iter in self.coeff_map:
            new_expr.coeff_map.pop(iter)
        return new_expr

    # replace an iterator with an affine idx expr
    def replace_iter(self, iter: Iter, idx_expr: "IndexExpr"):
        new_expr = self.copy()
        if iter not in self.coeff_map:
            return new_expr

        coeff = self.coeff_map[iter]

        # remove iterator
        new_expr.coeff_map.pop(iter)
        
        # update coeff map
        for i, c in idx_expr.coeff_map.items():
            if i not in new_expr.coeff_map:
                new_expr.coeff_map[i] = 0
            new_expr.coeff_map[i] += coeff * c
        
        # update constant term
        new_expr.const_term += coeff * idx_expr.const_term
        
        return new_expr


class MultiDimIndexExpr:
    """
    affine multi-dim index expressions
    e.g. [i*2, i*3+j*4+2], where i and j are Iterators
    """
    def __init__(self, idx_exprs: "list[IndexExpr]", morphable=False):
        super().__init__()
        self.idx_exprs = idx_exprs  # idx exprs of every dims
        self.morphable = morphable  # TODO: remove this later

    @classmethod
    def new(cls, idx_exprs, morphable=False):
        if morphable:  # idx_exprs: list[Iter]
            return cls([
                IndexExpr(OrderedDict([(it, 1)]))
                for it in idx_exprs
            ], morphable=True)
        else:  # idx_exprs: list[tuple[list[tuple], int]]
            return cls([
                IndexExpr(OrderedDict(coeffs), const)
                for coeffs, const in idx_exprs
            ], morphable=False)

    @property
    def ndim(self):
        return len(self.idx_exprs)

    @property
    def iters(self):
        return set().union(*[expr.iters for expr in self.idx_exprs])

    @property
    def spatial_iters(self):
        return (it for it in self.iters if it.iter_kind == "spatial")

    def __repr__(self):
        return ", ".join(list(map(str, self.idx_exprs)))

    def copy(self):
        return MultiDimIndexExpr([expr.copy() for expr in self.idx_exprs])

    # TODO: move into Step classes
    """ index transforms """

    # delete an iterator
    def delete_iter(self, iter: Iter):
        if self.morphable:  # ndim will change: need special handling
            dim_id = [iter in expr.coeff_map for expr in self.idx_exprs].index(True)
            idx_exprs = [idx_expr.copy() for idx_expr in self.idx_exprs]
            idx_exprs = idx_exprs[:dim_id] + idx_exprs[dim_id+1:]
            return MultiDimIndexExpr(idx_exprs, morphable=self.morphable)
        else:
            return MultiDimIndexExpr([
                idx_expr.delete_iter(iter)
                for idx_expr in self.idx_exprs
            ])

    # replace an iterator with an affine expr
    def replace_iter(self, iter: Iter, idx_expr: IndexExpr):
        assert not self.morphable
        return MultiDimIndexExpr([
            expr.replace_iter(iter, idx_expr)
            for expr in self.idx_exprs
        ])

    def split_iter(self, iter: Iter, spl_iters: "list[Iter]"):
        assert self.morphable
        dim_id = [iter in expr.coeff_map for expr in self.idx_exprs].index(True)
        spl_exprs = [IndexExpr(OrderedDict({iter: 1})) for iter in spl_iters]
        idx_exprs = [idx_expr.copy() for idx_expr in self.idx_exprs]
        idx_exprs = idx_exprs[:dim_id] + spl_exprs + idx_exprs[dim_id+1:]
        return MultiDimIndexExpr(idx_exprs, morphable=self.morphable)

    def add_iter(self, iter, dim_id=None, stride=None):
        idx_exprs = [idx_expr.copy() for idx_expr in self.idx_exprs]
        if self.morphable: 
            assert dim_id is None and stride is None
            idx_new_dim = IndexExpr(OrderedDict({iter: 1}))
            idx_exprs.append(idx_new_dim)
            return MultiDimIndexExpr(idx_exprs, morphable=self.morphable)
        else:
            assert dim_id is not None and stride is not None
            assert iter not in idx_exprs[dim_id].coeff_map
            idx_exprs[dim_id].coeff_map[iter] = stride
            return MultiDimIndexExpr(idx_exprs, morphable=self.morphable)


# TODO: support named dimensions
class Tensor:
    def __init__(self, name: str, init_shape: "list[int]", morphable: bool = False):
        self.name = name
        self.init_shape = init_shape
        self.shape = init_shape
        self.morphable = morphable
        self.accesses = list()  # TODO: find a better way to manage related ``TensorAccess``es

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return np.prod(self.shape)

    @property
    def stride(self):
        return list(reversed([1] + np.cumprod(list(reversed(self.shape))).tolist()[:-1]))

    @property
    def is_weight(self):
        return self.morphable and len(self.accesses) == 1

    def __repr__(self):
        return self.name

    def copy(self):
        return Tensor(self.name, self.shape.copy(), self.morphable)

    def set_morphable(self, morphable: bool):
        self.morphable = morphable
        for access in self.accesses:
            access.md_idx_expr.morphable = morphable

    # TODO: move into Step classes
    """ tensor transforms """

    def split_dim(self, dim_id, lengths):
        dim_size = self.shape[dim_id]
        assert dim_size % np.prod(lengths) == 0
        new_dims = [dim_size // np.prod(lengths), *lengths]
        self.shape = self.shape[:dim_id] + new_dims + self.shape[dim_id+1:]
        return self

    def elim_dim(self, dim_id):
        self.shape.pop(dim_id)
        return self

    def add_dim(self, length, dim_id=None):
        dim_id = len(self.shape) if dim_id is None else dim_id
        self.shape.insert(dim_id, length)
        return self


class TensorAccess:
    def __init__(self, tensor: Tensor, md_idx_expr: MultiDimIndexExpr):
        self.tensor = tensor
        self.tensor.accesses.append(self)
        self.md_idx_expr = md_idx_expr
        self.md_idx_expr.morphable = tensor.morphable
        assert self.tensor.ndim == self.md_idx_expr.ndim
        self.compute_expr: Optional[ComputeExpr] = None

    @classmethod
    def new(cls, tensor, md_idx):
        idx_expr = MultiDimIndexExpr.new(md_idx, tensor.morphable)
        return cls(tensor, idx_expr)

    @classmethod
    def new_init(cls, tensor, iters):
        md_idx = iters if tensor.morphable else [([(it, 1)], 0) for it in iters]
        return cls.new(tensor, md_idx)

    @property
    def morphable(self):
        return self.tensor.morphable

    @property
    def is_weight(self):
        return self.tensor.is_weight

    def __repr__(self) -> str:
        return str(self.tensor) + "[" + str(self.md_idx_expr) + "]"

    def copy(self, tensor_map):
        return TensorAccess(tensor_map[self.tensor], self.md_idx_expr.copy())

    def detach(self):
        self.tensor.accesses.remove(self)


class ComputeExpr:
    """
    einsum style compute expression
    A[i, j] = B[i, k] * C[k, i]
    """
    def __init__(self, output: TensorAccess, operands: "list[TensorAccess]"):
        # TODO: support compute expr with more operands
        assert len(operands) == 2
        
        self.output = output
        self.operands = operands
        
        for access in [output, *operands]:
            access.compute_expr = self
        
        self.stage: Optional[Stage] = None

    @property
    def spatial_iters(self):
        return set().union(*[acc.md_idx_expr.spatial_iters for acc in self.operands])

    @property
    def iters(self):
        return set().union(*[acc.md_idx_expr.iters for acc in self.accesses])

    @property
    def accesses(self):
        return [self.output, *self.operands]

    def __repr__(self):
        return str(self.output) + " = " + " * ".join(list(map(str, self.operands)))

    def copy(self, tensor_map):
        new_output = self.output.copy(tensor_map)
        new_operands = [acc.copy(tensor_map) for acc in self.operands]
        return ComputeExpr(new_output, new_operands)


class Stage: 
    def __init__(self, iters: "list[Iter]", compute_expr: ComputeExpr):
        self.iters = iters
        self.compute_expr = compute_expr
        assert self.compute_expr.iters == set(self.iters)
        self.compute_expr.stage = self

    @property
    def tensors(self):
        return list(set([tacc.tensor for tacc in self.compute_expr.accesses]))

    def copy(self, tensor_map):
        return Stage(self.iters, self.compute_expr.copy(tensor_map))
