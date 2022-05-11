from dataclasses import dataclass
from weakref import WeakValueDictionary

from functorch._C import (
    _add_batch_dim,
    _vmap_add_layers,
    _vmap_remove_layers,
    maybe_get_bdim,
    maybe_get_level,
    get_unwrapped,
    current_level,
)

from contextlib import contextmanager
import functorch


_levels = []
# in C this can just be a raw array without refcounting
_level_to_dim = WeakValueDictionary()

_enabled = False
@contextmanager
def _enable_layers(dims):
    global _enabled
    assert not _enabled
    input = list(sorted((d._level, d.size) for d in dims if not isinstance(d, int)))
    n = len(input)
    try:
        #print("BEGIN ", dims, input)
        #import pdb; pdb.set_trace()
        _vmap_add_layers(input)
        _enabled = True
        yield
        #print("FINISHED ", dims)
    finally:
        _enabled = False
        _vmap_remove_layers(n)

def _add_batch_dims(t, levels):
    from . import _Tensor
    assert not isinstance(t, _Tensor)
    levels = list(levels)
    for l in sorted((l for l in levels if not isinstance(l, int)), key=lambda x: x._level):
        i = levels.index(l)
        t = _add_batch_dim(t, i, l._level)
        del levels[i]
    return t

def _remove_batch_dims(t):
    levels = list(range(-t.ndim, 0))
    while True:
        l = maybe_get_level(t)
        if l == -1:
            break
        d = maybe_get_bdim(t)
        levels.insert(d, _level_to_dim[l])
        t = get_unwrapped(t)
    return t, levels



def _alloc_level(dim):
    assert len(_levels) <= 32
    dim._level = 31 + len(_levels)
    _level_to_dim[dim._level] = dim
    _levels.append(True)

def _free_level(dim):
    idx = dim._level - 31
    _levels[idx] = False
    while _levels and not _levels[-1]:
        _levels.pop()
