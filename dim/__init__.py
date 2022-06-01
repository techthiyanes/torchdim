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
    def ndim(self):
        return self._batchtensor.ndim

    @property
    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))

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

    def gather(self, dims, values):
        if isinstance(dims, int):
            return self.__torch_function__(torch.Tensor.gather, None, (self, dims, values))

        dims = _dims(dims, None, False, False)
        if not isinstance(values, (list, tuple, DimList)):
            values = (values,)
        add_dim = any(isinstance(v, TensorLike) and v.ndim == 0 for v in values)
        if add_dim: # add/remove fake dimension to trick advanced indexing
            values = tuple(v[None] for v in values)
        r = self.positional(*dims)[values]
        if add_dim:
            r = r[0]
        return r

    def _block(self, f: 'Dim', t: 'Sequence[Dim]'):
        #  splitting a dimension: f - Dim, t - List[Dim]
        _bind_one_dim(f, t)
        if f not in self.dims:
            raise DimensionMismatchError(f"tensor ({self.dims}) does not have dim: {f}")
        ptensor, levels = self._tensor, list(self._levels)
        f_idx = levels.index(f)
        new_sizes = list(ptensor.size())
        levels[f_idx:f_idx+1] = t
        new_sizes[f_idx:f_idx+1] = [e.size for e in t]

        return Tensor.from_positional(ptensor.view(*new_sizes), levels, self._has_device)

    def _flatten(self, f: 'Sequence[Dim]', t: 'Dim'):
        _bind_one_dim(t, f)
        ptensor, levels = self._tensor, list(self._levels)
        indices = tuple(levels.index(e) for e in f)
        start_idx = min(indices)
        end_idx = start_idx + len(indices)
        ptensor = ptensor.movedim(indices, tuple(range(start_idx, end_idx)))
        ptensor = ptensor.flatten(start_idx, end_idx - 1)
        for idx in sorted(indices, reverse=True):
            del levels[idx]
        levels.insert(start_idx, t)
        return Tensor.from_positional(ptensor, levels, self._has_device)

    def reshape_dim(self, f: 'Union[Dim, Sequence[Dim]]', t: 'Union[Dim, Sequence[Dim]]'):
        if not isinstance(f, Dim):
            if not isinstance(t, Dim):
                if isinstance(t, DimList) and not t.is_bound:
                    t.bind_len(len(f))
                if isinstance(f, DimList) and not f.is_bound:
                    f.bind_len(len(t))
                for a,b in zip(f, t):
                    self = self.reshape(a, b)
                return self
            else:
                return self._flatten(f, t)
        else:
            return self._block(f, [t] if isinstance(t, Dim) else t)

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
    return stack(tensors, n, dim).reshape_dim((n, dim), new_dim)

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

def _bind_dims_to_size(lhs_size, rhs, lhs_debug):
    not_bound = tuple((i, r) for i, r in enumerate(rhs) if not r.is_bound)
    if len(not_bound) == 1:
        idx, d = not_bound[0]
        rhs_so_far = prod(r.size for r in rhs if r.is_bound)
        if lhs_size % rhs_so_far != 0:
            raise DimensionMismatchError(f"inferred dimension does not evenly fit into larger dimension: {lhs_size} vs {tuple('?' if not r.is_bound else str(r.size) for r in rhs)}")
        new_size = lhs_size // rhs_so_far
        d.size = new_size
    elif len(not_bound) > 1:
        raise DimensionMismatchError(f"cannot infer the size of two dimensions at once: {rhs} with sizes {tuple('?' if not r.is_bound else str(r.size) for r in rhs)}")
    else:
        rhs_size = prod(r.size for r in rhs)
        if lhs_size != rhs_size:
            raise DimensionMismatchError(f"Dimension sizes to do not match ({lhs_size} != {rhs_size}) when matching {lhs_debug} to {rhs}")

def _bind_one_dim(lhs: 'Dim', rhs: 'Sequence[Dim]'):
    if not lhs.is_bound:
        lhs.size = prod(r.size for r in rhs)
    else:
        _bind_dims_to_size(lhs.size, rhs, lhs)


def _wrap_dim(d, N, keepdim):
    if isinstance(d, Dim):
        assert not keepdim, "cannot preserve first-class dimensions with keepdim=True"
        return d
    elif d >= 0:
        return d - N
    else:
        return d

def _tensor_levels(inp):
    if isinstance(inp, _Tensor):
        return inp._tensor, list(inp._levels), inp._has_device
    else:
        return inp, list(range(-inp.ndim, 0)), True

def _match_levels(v, from_levels, to_levels):
    view = []
    permute = []
    requires_view = False
    size = v.size()
    for t in to_levels:
        try:
            idx = from_levels.index(t)
            permute.append(idx)
            view.append(size[idx])
        except ValueError:
            view.append(1)
            requires_view = True
    if permute != list(range(len(permute))):
        v = v.permute(*permute)
    if requires_view:
        v = v.view(*view)
    return v

def _dims(d, N, keepdim, single_dim):
    if isinstance(d, (Dim, int)):
        return (_wrap_dim(d, N, keepdim),)
    assert not single_dim, f"expected a single dimension or int but found: {d}"
    return tuple(_wrap_dim(x, N, keepdim) for x in d)


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
