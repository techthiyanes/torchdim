import torch
from .batch_tensor import _enable_layers
from typing import Union, Sequence
import inspect
import dis
from .tree_map import tree_flatten, tree_map
from collections import defaultdict
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

class _Tensor:
    # fast path around slow wrapping/unwrapping logic for simply queries used
    # by the implementation...
    @property
    def ndim(self):
        return self._batchtensor.ndim

    @property
    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))

    if True:
        __torch_function__ = classmethod(_C.__torch_function__)
    else:
        @classmethod
        def __torch_function__(self, orig, cls, args, kwargs={}):
            if orig is torch.Tensor.__mul__:
                lhs, rhs = args
                if isinstance(lhs, _Tensor) and isinstance(rhs, _Tensor) and lhs.ndim == 0 and rhs.ndim == 0:
                    #print("END", orig)
                    return DelayedMulTensor(lhs, rhs)
            all_dims = []
            flat_args, unflatten = tree_flatten((args, kwargs))
            device_holding_tensor = None
            for f in flat_args:
                if isinstance(f, _Tensor):
                    if f._has_device:
                        device_holding_tensor = f._batchtensor
                    for d in f.dims:
                        if d not in all_dims:
                            all_dims.append(d)
            def unwrap(t):
                if isinstance(t, _Tensor):
                    r = t._batchtensor
                    if device_holding_tensor is not None and not t._has_device:
                        r = r.to(device=device_holding_tensor.device)
                    return r
                return t

            if orig in pointwise:
                result_levels = []
                arg_levels = []
                to_expand = []
                for i,f in enumerate(flat_args):
                    if isinstance(f, TensorLike):
                        ptensor, levels, _ = _tensor_levels(f)
                        if isinstance(f, _Tensor) and not f._has_device and device_holding_tensor is not None:
                            ptensor = ptensor.to(device=device_holding_tensor.device)
                        flat_args[i] = ptensor
                        for l in levels:
                            if l not in result_levels:
                                result_levels.append(l)
                        to_expand.append((i, levels))

                for i, levels in to_expand:
                    flat_args[i] = _match_levels(flat_args[i], levels, result_levels)
                args, kwargs = unflatten(flat_args)
                result = orig(*args, **kwargs)
                def wrap(t):
                    if isinstance(t, TensorLike):
                        return Tensor.from_positional(t, result_levels, device_holding_tensor is not None)
                    return t
                return tree_map(wrap, result)
            else:
                def wrap(t):
                    if isinstance(t, TensorLike):
                        return Tensor.from_batched(t, device_holding_tensor is not None)
                    return t
                with _enable_layers(all_dims):
                    print(f"batch_tensor for {orig}")
                    args, kwargs = unflatten(unwrap(f) for f  in flat_args)
                    result = orig(*args, **kwargs)
                    # print("END", orig)
                    return tree_map(wrap, result)

    def __repr__(self):
        tensor, levels = self._tensor, self._levels
        return f'{tensor}\nwith dims={levels} {tensor.size()}'
    if True:
        positional = _C._instancemethod(_C.positional)
    else:
        def positional(self, *dims):
            ptensor, levels = self._tensor, list(self._levels)
            flat_dims = []
            view = []
            needs_view = False
            for d in dims:
                if isinstance(d, DimList):
                    flat_dims.extend(d)
                    view.extend(e.size for e in d)
                elif isinstance(d, Dim):
                    flat_dims.append(d)
                    view.append(d.size)
                else:
                    flat_dims.extend(d)
                    view.append(prod(e.size for e in d))
                    needs_view = True

            permute = list(range(len(levels)))
            ndim = self.ndim
            nflat = len(flat_dims)
            for i, d in enumerate(flat_dims):
                try:
                    idx = levels.index(d)
                except ValueError as e:
                    raise DimensionBindError(f'tensor of dimensions {self.dims} does not contain dim {d}') from e
                p = permute[idx]
                del levels[idx]
                del permute[idx]
                levels.insert(i, -ndim - (nflat - i))
                permute.insert(i, p)
            ptensor = ptensor.permute(*permute)
            result = Tensor.from_positional(ptensor, levels, self._has_device)
            if needs_view:
                result = result.reshape(*view, *result.size()[len(flat_dims):])
            return result

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

    # ops that should also just take dims as arguments
    # reductions - important ones implemented (for the dim argument)
    # unstack - implemented as unbind
    # stack - implemented

    # these are overloads of the original API
    #
    if True:
        expand = _C._instancemethod(_C.expand)
    else:
        def expand(self, *sizes):
            if not _contains_dim(sizes):
                return self.__torch_function__(torch.Tensor.expand, None, (self, *sizes))
            dims = sizes
            sizes = [d.size for d in dims] + [-1]*self.ndim
            self = self.expand(*sizes)
            return self[dims]

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

if True:
    _wrap = _C._wrap
    def _def(name, *args, **kwargs):
        orig = getattr(torch.Tensor, name)
        setattr(_Tensor, name, _C._instancemethod(_wrap(orig, *args, **kwargs)))
else:
    _not_present = object()

    def _getarg(name, offset, args, kwargs, default):
        if len(args) > offset:
            return args[offset]
        return kwargs.get(name, default)

    def _patcharg(name, offset, args, kwargs, value):
        if len(args) > offset:
            args[offset] = value
        else:
            kwargs[name] = value

    def _wrap(orig, dim_offset=0, keepdim_offset=1, dim_name='dim', single_dim=False, reduce=True):
        def fn(self, *args, **kwargs):
            dim = _getarg(dim_name, dim_offset, args, kwargs, _not_present)
            if dim is _not_present or (single_dim and not isinstance(dim, Dim)):
                with _enable_layers(self.dims):
                    print(f"dim fallback batch_tensor for {orig}")
                    return Tensor.from_batched(orig(self._batchtensor, *args, **kwargs), self._has_device)
            keepdim = _getarg('keepdim', keepdim_offset, args, kwargs, False) if reduce else False
            t, levels = self._tensor, list(self._levels)
            dims = _dims(dim, self._batchtensor.ndim, keepdim, single_dim)
            dim_indices = tuple(levels.index(d) for d in dims)
            if reduce and not keepdim:
                new_levels = [l for i, l in enumerate(levels) if i not in dim_indices]
            else:
                new_levels = levels

            if len(dim_indices) == 1:
                dim_indices = dim_indices[0] # so that dims that really only take a single argument work...
            args = list(args)
            _patcharg(dim_name, dim_offset, args, kwargs, dim_indices)
            def wrap(t):
                if isinstance(t, TensorLike):
                    return Tensor.from_positional(t, new_levels, self._has_device)
                return t
            with _enable_layers(new_levels):
                print(f"dim used batch_tensor for {orig}")
                r = orig(t, *args, **kwargs)
                return tree_map(wrap, r)
        return fn

    def _def(name, *args, **kwargs):
        orig = getattr(torch.Tensor, name)
        setattr(_Tensor, name, _wrap(orig, *args, **kwargs))


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


class Dim(_C.Dim, _Tensor):
    # Tensor defines these methods for actual tensor data
    # we want dims to behave like individual objects for
    # hashing and printing, so we revert back to the object implementation
    __format__ = object.__format__
    __hash__ = object.__hash__

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

class Tensor(_Tensor, _C.Tensor):
    from_batched = staticmethod(_C.Tensor_from_batched)
    from_positional = staticmethod(_C.Tensor_from_positional)
    sum = _C._instancemethod(_C.Tensor_sum)

class DelayedMulTensor(_Tensor):
    def __init__(self, lhs, rhs):
        self._lhs, self._rhs = lhs, rhs
        self._data = None
        self._levels_data = None
        self._has_device = lhs._has_device or rhs._has_device

    @property
    def _levels(self):
        if self._levels_data is None:
            levels = list(self._lhs._levels)
            for l in self._rhs._levels:
                if l not in levels:
                    levels.append(l)
            self._levels_data = tuple(levels)
        return self._levels_data

    @property
    def _batchtensor(self):
        if self._batchtensor_data is None:
            with _enable_layers(self._levels):
                print(f"bt multiply fallback")
                self._batchtensor_data = self._lhs._batchtensor * self._rhs._batchtensor
        return self._batchtensor_data

    @property
    def _tensor(self):
        raise NotImplementedError()

    def sum(self, dim):
        dims = _dims(dim, 0, False, False)
        n = ord('a')
        all_levels = self._levels
        def to_char(d):
            return chr(n + all_levels.index(d))
        plhs, levelslhs = self._lhs._tensor, self._lhs._levels
        prhs, levelsrhs = self._rhs._tensor, self._rhs._levels
        new_dims = tuple(d for d in self.dims if d not in dims)
        new_levels = [l for l in self._levels if l not in dims]
        fmt = ''.join([*(to_char(d) for d in levelslhs), ',', *(to_char(d) for d in levelsrhs), '->', *(to_char(d) for d in new_levels)])
        result_data = torch.einsum(fmt, (plhs, prhs))
        return Tensor.from_positional(result_data, new_levels, True)



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

_orig_getitem = torch.Tensor.__getitem__
def _contains_dim(input):
    for i in input:
        if isinstance(i,  Dim):
            return True

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

no_slice = slice(None)
if True:
    __getitem__ = _C._instancemethod(_C.__getitem__)
    __setitem__ = _C._instancemethod(_C.__setitem__)
else:
    def __getitem__(self, input):
        # * bail to original example if we have a single non-Dim tensor, or a non-tensor
        # * locate ... or an unbound tensor list, and determine its size, bind dim list
        #   (remember that None does not count to the total dim count)
        # * bind simple dims and dim-packs to their sizes, count the number of uses of each dim,
        #   produce the re-view if needed
        # * for each single-use dim index, replace with no_slice and mark that it will be added
        #   (keep track of whether we have to call super)
        # * call super if needed
        # * if we have dims to bind, bind them (it will help if we eliminated ... and None before)

        # this handles bool indexing handling, as well as some other simple cases.
        if (not isinstance(input, Dim) and
            not isinstance(input, tuple) and
            # WAR for functorch bug where zero time tensors in getitem are not handled correctly.
            not (isinstance(input, TensorLike) and input.ndim == 0)):
                if isinstance(self, _Tensor):
                    return _Tensor.__torch_function__(_orig_getitem, None, (self, input))
                else:
                    return _orig_getitem(self, input)

        # can further optimize this case
        if not isinstance(input, tuple):
            input = [input]
        else:
            input = list(input)

        dims_indexed = 0
        expanding_object = None
        dimlists = []
        for i, s in enumerate(input):
            if s is ... or isinstance(s, DimList) and not s.is_bound:
                if expanding_object is not None:
                    raise DimensionBindError(f'at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets {i} and {expanding_object}')
                expanding_object = i

            if isinstance(s, DimList):
                dims_indexed += len(s) if s.is_bound else 0
                dimlists.append(i)
            elif s is not None and s is not ...:
                dims_indexed += 1

        ndim = self.ndim
        if dims_indexed > ndim:
            raise IndexError(f'at least {dims_indexed} indices were supplied but the tensor only has {ndim} dimensions.')
        if expanding_object is not None:
            expanding_ndims = ndim - dims_indexed
            obj = input[expanding_object]
            if obj is ...:
                input[expanding_object:expanding_object+1] = [no_slice]*expanding_ndims
            else:
                obj.bind_len(expanding_ndims)
        # flatten the dimslists into the indexing
        for i in reversed(dimlists):
            input[i:i+1] = input[i]
        dims_indexed = 0
        requires_view = False
        size = self.size()
        view_sizes = []
        dims_seen = defaultdict(lambda: 0)

        def add_dims(t):
            if not isinstance(t, _Tensor):
                return
            for d in t.dims:
                dims_seen[d] += 1

        add_dims(self)
        dim_packs = []
        for i, idx in enumerate(input):
            if idx is None:
                input[i] = no_slice
                view_sizes.append(1)
                requires_view = True
            else:
                sz = size[dims_indexed]
                if isinstance(idx, Dim):
                    idx.size = sz
                    dims_seen[idx] += 1
                    view_sizes.append(sz)
                elif isinstance(idx, tuple) and idx and isinstance(idx[0], Dim):
                    for d in idx:
                        dims_seen[d] += 1
                    _bind_dims_to_size(sz, idx, f'offset {i}')
                    view_sizes.extend(d.size for d in idx)
                    requires_view=True
                    dim_packs.append(i)
                else:
                    add_dims(idx)
                    view_sizes.append(sz)
                dims_indexed += 1
        if requires_view:
            self = self.view(*view_sizes)
        for i in reversed(dim_packs):
            input[i:i+1] = input[i]

        # currenty:
        # input is flat, containing either Dim, or Tensor, or something valid for standard indexing
        # self may have first-class dims as well.

        # to index:
        # drop the first class dims from self, they just become direct indices of their positions

        # figure out the dimensions of the indexing tensors: union of all the dims in the tensors in the index.
        # these dimensions will appear and need to be bound at the first place tensor occures

        if isinstance(self, _Tensor):
            ptensor_self, levels = self._tensor, list(self._levels)
            # indices to ptensor rather than self which has first-class dimensions
            input_it = iter(input)
            flat_inputs = [next(input_it) if isinstance(l, int) else l for l in levels]
            has_device = self._has_device
            to_pad = 0
        else:
            ptensor_self, flat_inputs = self, input
            to_pad = ptensor_self.ndim - len(flat_inputs)
            has_device = True

        result_levels = []
        index_levels = []
        tensor_insert_point = None
        to_expand = {}
        requires_getindex = False
        for i, inp in enumerate(flat_inputs):
            if isinstance(inp, Dim) and dims_seen[inp] == 1:
                flat_inputs[i] = no_slice
                result_levels.append(inp)
            elif isinstance(inp, TensorLike):
                requires_getindex = True
                if tensor_insert_point is None:
                    tensor_insert_point = len(result_levels)
                ptensor, levels, _ = _tensor_levels(inp)
                to_expand[i] = levels
                flat_inputs[i] = ptensor
                for l in levels:
                    if l not in index_levels:
                        index_levels.append(l)
            else:
                requires_getindex = True
                result_levels.append(0)

        if tensor_insert_point is not None:
            result_levels[tensor_insert_point:tensor_insert_point] = index_levels

        for i, levels in to_expand.items():
            flat_inputs[i] = _match_levels(flat_inputs[i], levels, index_levels)

        if requires_getindex:
            result = _orig_getitem(ptensor_self, flat_inputs)
        else:
            result = ptensor_self

        next_positional = -1
        if to_pad > 0:
            result_levels.extend([0]*to_pad)
        for i, r in enumerate(reversed(result_levels)):
            if isinstance(r, int):
                result_levels[-1 - i] = next_positional
                next_positional -= 1

        return Tensor.from_positional(result, result_levels, has_device)


torch.Tensor.__getitem__ = __getitem__
_Tensor.__getitem__ = __getitem__
torch.Tensor.__setitem__ = __setitem__
_Tensor.__setitem__ = __setitem__

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
wrap_type(_Tensor, torch.Tensor, _Tensor.__torch_function__)
