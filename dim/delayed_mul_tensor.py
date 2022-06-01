import torch
from . import _Tensor, _dims, Tensor

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
