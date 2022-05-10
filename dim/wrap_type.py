import torch
from types import FunctionType, BuiltinMethodType, MethodDescriptorType, WrapperDescriptorType, GetSetDescriptorType
from pprint import pprint
from dim._C import _wrap_method

FUNC_TYPES = (FunctionType, MethodDescriptorType, BuiltinMethodType, WrapperDescriptorType)
PROPERTY_TYPES = (GetSetDescriptorType,property)

# def _wrap_method(orig, __torch_function__):
#     def impl(*args, **kwargs):
#         return __torch_function__(orig, None, args, kwargs)
#     return impl


def _wrap_attr(orig, __torch_function__):
    return property(_wrap_method(orig.__get__, __torch_function__))

def wrap_type(to_patch, pattern, __torch_function__):
    all = {}
    for t in reversed(pattern.mro()[:-1]): # skip object
        all.update(t.__dict__)

    for name, obj in all.items():
        if name in ('__dict__','__new__', '__init__', '__repr__', '__weakref__', '__doc__', '__module__'):
            continue

        # skip things that have been overloaded
        # things that come from object like `__eq__` still need to be patched, however.
        if hasattr(to_patch, name) and getattr(to_patch, name) is not getattr(object, name, None):
            continue

        if isinstance(obj, FUNC_TYPES):
            setattr(to_patch, name, _wrap_method(obj, __torch_function__))
        elif isinstance(obj, PROPERTY_TYPES):
            setattr(to_patch, name, _wrap_attr(obj, __torch_function__))
