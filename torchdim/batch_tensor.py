from functorch._C import (
    _vmap_add_layers,
    _vmap_remove_layers,
)

from contextlib import contextmanager

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
