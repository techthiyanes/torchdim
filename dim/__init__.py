import torch
from .batch_tensor import _enable_layers
from typing import Union, Sequence
import inspect
import dis
from .tree_map import tree_flatten, tree_map
from .wrap_type import wrap_type
import dim._C as _C
from dim._C import dims, DimList


class DimensionMismatchError(Exception):
    pass

class DimensionBindError(Exception):
    pass

from functools import reduce
import operator
from . import op_properties

prod = lambda x: reduce(operator.mul, x, 1)

# use dict to avoid writing C++ bindings for set
pointwise = {t: True for t in op_properties.pointwise}

use_c = True
if not use_c:
    from . import reference

class _Tensor:
    # fast path around slow wrapping/unwrapping logic for simply queries used
    # by the implementation...


    @property
    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))

    def dim(self):
        return self.ndim

    if use_c:
        __torch_function__ = classmethod(_C.__torch_function__)
        positional = _C._instancemethod(_C.positional)
        expand = _C._instancemethod(_C.expand)
    else:
        __torch_function__ = reference.__torch_function__
        positional = reference.positional
        expand = reference.expand

    index = _C._instancemethod(_C.index)

    def __repr__(self):
        tensor, levels = self._tensor, self._levels
        return f'{tensor}\nwith dims={levels} {tensor.size()}'

    # make a single dimension positional but do not permute it,
    # used to do multi-tensor operators where the dim being acted on
    # should not physically move if possible
    def _positional_no_permute(self, dim, expand_dim=False):
        ptensor, levels = self._tensor, list(self._levels)
        try:
            idx = levels.index(dim)
        except ValueError:
            if not expand_dim:
                raise
            idx = 0
            ptensor = ptensor.expand(dim.size, *ptensor.size())
            levels.insert(0, 0)
        idx_batched = 0
        for i in range(idx):
            if isinstance(levels[i], int):
                levels[i] -= 1
                idx_batched += 1
        levels[idx] = -idx_batched - 1
        return Tensor.from_positional(ptensor, levels, self._has_device), idx_batched


TensorLike = (_Tensor, torch.Tensor)

class Dim(_C.Dim, _Tensor):
    # Tensor defines these methods for actual tensor data
    # we want dims to behave like individual objects for
    # hashing and printing, so we revert back to the object implementation
    __format__ = object.__format__
    __hash__ = object.__hash__

class Tensor(_Tensor, _C.Tensor):
    from_batched = staticmethod(_C.Tensor_from_batched)
    from_positional = staticmethod(_C.Tensor_from_positional)
    sum = _C._instancemethod(_C.Tensor_sum)

# XXX - dim is optional and can be the outer-most dimension...
def stack(tensors, new_dim, dim=0, out=None):
    if isinstance(dim, int):
        return _bind(torch.stack(tensors, dim, out), (dim,), (new_dim,))
    index = None
    if out is not None:
        out, index = out._positional_no_permute(dim, expand_dim=True)
    ptensors = []
    for t in tensors:
        pt, pi = t._positional_no_permute(dim, expand_dim=True)
        if index is not None and pi != index:
            pt = pt.move_dim(pi, index)
        else:
            index = pi
        ptensors.append(pt)
    pr = torch.stack(ptensors, index, out=out)
    return _bind(pr, (index, index + 1), (new_dim, dim))

def cat(tensors, dim, new_dim):
    n = dims()
    return stack(tensors, n, dim).index([n, dim], new_dim)

if use_c:
    _wrap = _C._wrap
    def _def(name, *args, **kwargs):
        orig = getattr(torch.Tensor, name)
        setattr(_Tensor, name, _C._instancemethod(_wrap(orig, *args, **kwargs)))
    t__getitem__ = _C._instancemethod(_C.__getitem__)
else:
    _wrap, _def = reference._wrap, reference._def
    t__getitem__ = reference.t__getitem__

# note: there is not python reference
t__setitem__ = _C._instancemethod(_C.__setitem__)

def _tensor_levels(inp):
    if isinstance(inp, _Tensor):
        return inp._tensor, list(inp._levels), inp._has_device
    else:
        return inp, list(range(-inp.ndim, 0)), True


def _bind(self, offset, dims):
    ptensor, levels, has_device = _tensor_levels(self)
    next_idx = 0
    for i, (l, sz) in enumerate(zip(levels, ptensor.size())):
        if isinstance(l, int):
            try:
                idx =offset.index(next_idx)
                d = levels[i] = dims[idx]
                d.size = sz
            except ValueError:
                pass
            next_idx += 1

    next_non_dim = -1
    for i in range(len(levels) - 1, -1, -1):
        if isinstance(levels[i], int):
            levels[i] = next_non_dim
            next_non_dim -= 1
    return Tensor.from_positional(ptensor, levels, has_device)


torch.Tensor.__getitem__ = t__getitem__
_Tensor.__getitem__ = t__getitem__
torch.Tensor.__setitem__ = t__setitem__
_Tensor.__setitem__ = t__setitem__

_orig_split = torch.Tensor.split
def split(self, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int) or any(isinstance(t, int) for t in split_size_or_sections):
        if isinstance(dim, Dim):
            raise ValueError(f'when dim is specified as a Dim object, split sizes must also be dimensions.')
        return _orig_split(self, split_size_or_sections, dim=dim)

    if isinstance(dim, Dim):
        assert isinstance(self, _Tensor), f"Tensor does not have dimension {dim}"
        self, dim = self._positional_no_permute(dim)

    size = self.size(dim)
    total_bound_size = 0
    unbound = []
    sizes = []
    for i, d in enumerate(split_size_or_sections):
        if d.is_bound:
            sizes.append(d.size)
            total_bound_size += d.size
        else:
            sizes.append(0)
            unbound.append(i)

    if unbound:
        assert total_bound_size <= size, f"result dimensions are larger than original: {total_bound_size} vs {size} ({split_size_or_sections})"
        remaining_size = size - total_bound_size
        chunk_size = -(-remaining_size // len(unbound))
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            split_size_or_sections[u].size = sz
            sizes[u] = sz
            remaining_size -= sz
    else:
        assert total_bound_size == size, f"result dimensions do not match original: {total_bound_size} vs {size} ({split_size_or_sections})"
    return tuple(_bind(t, (dim,), (d,)) for d, t in zip(split_size_or_sections, _orig_split(self, sizes, dim=dim)))

torch.Tensor.split = split
_Tensor.split = split
torch.Tensor.expand = _C._instancemethod(_C.expand)
torch.Tensor.index = _C._instancemethod(_C.index)
wrap_type(use_c, _Tensor, torch.Tensor, _Tensor.__torch_function__)

_def('mean')
_def('sum')
_def('all')
_def('amax')
_def('amin')
_def('aminmax')
_def('any')
_def('count_nonzero')
_def('logsumexp')
_def('nanmean')
_def('nansum')
_def('prod')
_def('std', keepdim_offset=2)
_def('var', keepdim_offset=2)
_def('max', single_dim=True)
_def('min', single_dim=True)
_def('argmax', single_dim=True)
_def('argmin', single_dim=True)
_def('kthvalue', single_dim=True)
_def('median', single_dim=True)
_def('nanmedian', single_dim=True)
_def('mode', single_dim=True)
_def('sort', reduce=False)
_def('argsort', reduce=False)
_def('unbind', single_dim=True)
_def('chunk', dim_offset=1, reduce=False)
_def('cummax', single_dim=True, reduce=False)
_def('cummin', single_dim=True, reduce=False)
_def('cumprod', single_dim=True, reduce=False)
_def('cumprod_', single_dim=True, reduce=False)
_def('cumsum', single_dim=True, reduce=False)
_def('cumsum_', single_dim=True, reduce=False)
_def('logcumsumexp', single_dim=True, reduce=False)
_def('renorm', dim_offset=1, single_dim=True, reduce=False)
_def('softmax', single_dim=True, reduce=False)
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# stuff to handle in the future, because they require special
# binding logic for dims
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (new dimes on output)
# should these all be subsumed by inplace indexing?
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
