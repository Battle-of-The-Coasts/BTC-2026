"""Torch-free reader for tensors saved with ``torch.save`` (zip format).

MGTAB and the standardized Cresci-15 release ship tensors as ``.pt`` files. Installing PyTorch only to read
them is heavy, so this module unpickles the archive and rebuilds each tensor as a NumPy array.
Only plain tensors (and dict/list containers of tensors) are supported.
"""
from __future__ import annotations

import collections
import io
import pickle
import zipfile

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "ByteStorage": np.uint8, "CharStorage": np.int8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


def load_pt(path: str):
    """Return the object stored in a ``torch.save`` archive with tensors replaced by NumPy arrays."""
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
                    (arr,) = storage
                    itemsize = arr.dtype.itemsize
                    view = np.lib.stride_tricks.as_strided(
                        arr[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
                    )
                    return np.array(view)  # contiguous copy, detached from the zip buffer
                return rebuild
            if name.endswith("Storage"):
                return _StorageType(name)
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, backward_hooks: data
            raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

        def persistent_load(self, pid):
            typename, storage_type, key, _location, _numel = pid
            if typename != "storage":
                raise pickle.UnpicklingError(f"unsupported persistent id {typename}")
            raw = zf.read(f"{prefix}/data/{key}")
            return (np.frombuffer(raw, dtype=_DTYPES[storage_type.name]),)

    return _Unpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
