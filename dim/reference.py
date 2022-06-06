# reference python implementations for C ops
import torch
from .tree_map import tree_flatten, tree_map
from .batch_tensor import _enable_layers
from . import op_properties
from dim._C import DimList
from collections import defaultdict
from functools import reduce
import operator


# use dict to avoid writing C++ bindings for set
pointwise = set(op_properties.pointwise)
prod = lambda x: reduce(operator.mul, x, 1)


def _wrap_dim(d, N, keepdim):
    from . import Dim
    if isinstance(d, Dim):
        assert not keepdim, "cannot preserve first-class dimensions with keepdim=True"
        return d
    elif d >= 0:
        return d - N
    else:
        return d

def _dims(d, N, keepdim, single_dim):
    from . import Dim
    if isinstance(d, (Dim, int)):
        return (_wrap_dim(d, N, keepdim),)
    assert not single_dim, f"expected a single dimension or int but found: {d}"
    return tuple(_wrap_dim(x, N, keepdim) for x in d)

def _bind_dims_to_size(lhs_size, rhs, lhs_debug):
    from . import DimensionMismatchError
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

def _tensor_levels(inp):
    from . import _Tensor
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


@classmethod
def __torch_function__(self, orig, cls, args, kwargs={}):
    from . import _Tensor, TensorLike, Tensor, _enable_layers
    from .delayed_mul_tensor import DelayedMulTensor

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

def positional(self, *dims):
    from . import Dim, Tensor
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

def _contains_dim(input):
    from . import Dim
    for i in input:
        if isinstance(i,  Dim):
            return True

def expand(self, *sizes):
    if not _contains_dim(sizes):
        return self.__torch_function__(torch.Tensor.expand, None, (self, *sizes))
    dims = sizes
    sizes = [d.size for d in dims] + [-1]*self.ndim
    self = self.expand(*sizes)
    return self[dims]


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
    from . import TensorLike, Dim, Tensor
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
    from . import _Tensor
    orig = getattr(torch.Tensor, name)
    setattr(_Tensor, name, _wrap(orig, *args, **kwargs))

no_slice = slice(None)

_orig_getitem = torch.Tensor.__getitem__

def t__getitem__(self, input):
    from . import Dim, DimensionBindError, _Tensor, TensorLike, DimList, Tensor
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
        not isinstance(input, (tuple,list)) and
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
            elif isinstance(idx, (tuple, list)) and idx and isinstance(idx[0], Dim):
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
