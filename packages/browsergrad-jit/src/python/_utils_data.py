"""browsergrad_jit.utils.data - Dataset + DataLoader.

Small, browser-safe subset of torch.utils.data. This deliberately mirrors the
eager browsergrad_grad implementation, with one JIT-specific difference:
TensorProxy refuses implicit NumPy conversion, so collation explicitly realizes
TensorProxy samples and wraps stacked batches back into TensorProxy objects.
"""

from __future__ import annotations

import math as _math
from typing import Any

import numpy as np


class Dataset:
    """Abstract base. Subclasses must override __len__ and __getitem__."""

    def __len__(self):
        raise NotImplementedError("Dataset subclasses must implement __len__")

    def __getitem__(self, index):
        raise NotImplementedError("Dataset subclasses must implement __getitem__")


def _is_tensor_proxy(value: Any) -> bool:
    try:
        from ._tensor_proxy import TensorProxy

        return isinstance(value, TensorProxy)
    except Exception:
        return False


def _as_array(value: Any) -> np.ndarray:
    if _is_tensor_proxy(value):
        return value.numpy()
    return np.asarray(value)


def _wrap_like(stacked: np.ndarray, sample: Any):
    if _is_tensor_proxy(sample):
        from ._tensor_proxy import from_numpy

        return from_numpy(np.asarray(stacked).copy())
    return stacked


def _collate_column(items):
    """Stack a homogeneous list of items into one TensorProxy or ndarray."""
    first = items[0]
    stacked = np.stack([_as_array(s) for s in items])
    return _wrap_like(stacked, first)


def _default_collate(samples):
    """Stack a list of samples into one batch.

    - If samples are tuples of the same arity, returns a tuple of stacked
      columns.
    - If a column contains TensorProxy samples, returns a TensorProxy batch.
    - Otherwise returns a NumPy array batch.
    """
    if len(samples) == 0:
        raise RuntimeError("_default_collate: empty batch")
    first = samples[0]
    if isinstance(first, tuple):
        arity = len(first)
        return tuple(_collate_column([s[i] for s in samples]) for i in range(arity))
    return _collate_column(samples)


class DataLoader:
    """Iterates a Dataset in single-process batches.

    Args:
        dataset: a Dataset or anything with __len__ and __getitem__.
        batch_size: number of samples per batch.
        shuffle: shuffle indices before each epoch.
        drop_last: skip the final partial batch.
        num_workers: must be 0 in browser/Pyodide.
        collate_fn: optional replacement for default tuple-aware collation.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn=None,
    ):
        if num_workers != 0:
            raise NotImplementedError(
                "DataLoader: num_workers > 0 is not supported in "
                "browsergrad_jit because Pyodide runs this loader in a "
                "single browser worker. Use num_workers=0."
            )
        if batch_size <= 0:
            raise ValueError(f"DataLoader: batch_size must be > 0, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn or _default_collate

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return _math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            indices = np.random.permutation(indices)
        bs = self.batch_size
        end = (n // bs) * bs if self.drop_last else n
        for start in range(0, end, bs):
            batch_idx = indices[start : start + bs]
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(samples)


class TensorDataset(Dataset):
    """Dataset wrapping multiple equal-length tensors, proxies, or ndarrays."""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        arrays = tuple(_as_array(t) for t in tensors)
        n = len(arrays[0])
        for arr in arrays[1:]:
            if len(arr) != n:
                raise ValueError(
                    "TensorDataset: all tensors must have the same first-dim size"
                )
        self.tensors = arrays
        self._returns_proxy = tuple(_is_tensor_proxy(t) for t in tensors)
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        out = []
        for arr, returns_proxy in zip(self.tensors, self._returns_proxy):
            item = np.asarray(arr[i])
            if returns_proxy:
                from ._tensor_proxy import from_numpy

                out.append(from_numpy(item.copy()))
            else:
                out.append(item)
        return tuple(out)


__all__ = ["Dataset", "DataLoader", "TensorDataset"]
