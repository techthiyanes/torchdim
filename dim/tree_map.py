from collections import namedtuple
from typing import NamedTuple

def tree_flatten_instance(r, obj):
    if isinstance(obj, dict):
        ctors = [tree_flatten_instance(r, v) for v in obj.values()]
        keys = list(obj.keys())
        return lambda n: {k: v(n) for k, v in zip(keys, ctors)}
    elif isinstance(obj, (list, tuple, NamedTuple)):
        t = type(obj)
        ctors = [tree_flatten_instance(r, v) for v in obj]
        return lambda n: t(ctor(n) for ctor in ctors)
    else:
        r.append(obj)
        return next

def tree_flatten(tree):
    r = []
    ctor = tree_flatten_instance(r, tree)
    return r, lambda ns: ctor(iter(ns))

def tree_map(fn, tree):
    vs, unflatten = tree_flatten(tree)
    return unflatten(fn(v) for v in vs)
