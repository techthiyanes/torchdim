from dataclasses import dataclass

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


_enabled = False
@contextmanager
def _enable_layers(dims):
    global _enabled
    assert not _enabled
    n = len(dims)
    try:
        input = sorted((-d.level, d.size) for d in dims)
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
    for l in sorted(levels, reverse=True):
        if l < 0:
            i = levels.index(l)
            # lvl = _vmap_levels[-l - 2]
            # assert lvl.level == l
            # assert lvl.size == t.size(i)
            t = _add_batch_dim(t, i, -l)
            del levels[i]
    return t

def _remove_batch_dims(t):
    levels = list(range(t.ndim))
    while True:
        l = maybe_get_level(t)
        if l == -1:
            break
        d = maybe_get_bdim(t)
        levels.insert(d, -l)
        t = get_unwrapped(t)
    return t, levels


_levels = []
def _alloc_level():
    assert len(_levels) <= 32
    r = -(31 + len(_levels))
    _levels.append(True)
    return r

def _free_level(l):
    idx = -l - 31
    _levels[idx] = False
    while _levels and not _levels[-1]:
        _levels.pop()
