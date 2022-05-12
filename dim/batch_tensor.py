from functorch._C import (
    _add_batch_dim,
    _vmap_add_layers,
    _vmap_remove_layers,
)
import functorch._C


from dim._C import _level_to_dim

from contextlib import contextmanager
import functorch


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
        print("Add batch dim ", i, l._level)
        t = _add_batch_dim(t, i, l._level)
        del levels[i]
    return t
