
"""browsergrad_grad.utils.data — Dataset + DataLoader."""

import numpy as np
import math as _bg_data_math


class Dataset:
    """Abstract base. Subclasses must override __len__ and __getitem__.

    Mirrors torch.utils.data.Dataset. Returning a tuple from __getitem__ is
    supported — DataLoader will stack each element separately.
    """
    def __len__(self):
        raise NotImplementedError("Dataset subclasses must implement __len__")
    def __getitem__(self, index):
        raise NotImplementedError("Dataset subclasses must implement __getitem__")


def _collate_column(items):
    """Stack a homogeneous list of items into one tensor or array."""
    try:
        from browsergrad_grad.tensor import Tensor as _T, stack as _stack
        if isinstance(items[0], _T):
            return _stack(list(items))
    except Exception:
        pass
    return np.stack([np.asarray(s) for s in items])


def _default_collate(samples):
    """Stack a list of samples into one batch.

    - If samples are tuples of the same arity, returns a tuple of stacks
      (one per position), each stack wrapped as a Tensor when the input items
      are Tensors.
    - Otherwise returns one stack.
    """
    if len(samples) == 0:
        raise RuntimeError("_default_collate: empty batch")
    first = samples[0]
    if isinstance(first, tuple):
        arity = len(first)
        return tuple(
            _collate_column([s[i] for s in samples])
            for i in range(arity)
        )
    return _collate_column(samples)


class DataLoader:
    """Iterates a Dataset in batches.

    Args:
        dataset: a Dataset (or anything with __len__ + __getitem__).
        batch_size: number of samples per batch.
        shuffle: shuffle indices before each epoch.
        drop_last: skip the final partial batch.
        num_workers: must be 0 in-browser. Anything else raises.
        collate_fn: how to stack samples; default stacks via np.stack and
            handles tuple outputs.

    Yields one batch per __next__. Re-iterable: a fresh shuffle happens on
    each call to __iter__.
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
                "DataLoader: num_workers > 0 is not supported in browsergrad_grad — "
                "Pyodide runs single-threaded. Use num_workers=0."
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
        return _bg_data_math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            indices = np.random.permutation(indices)
        bs = self.batch_size
        end = (n // bs) * bs if self.drop_last else n
        for start in range(0, end, bs):
            batch_idx = indices[start:start + bs]
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(samples)


class TensorDataset(Dataset):
    """Dataset wrapping multiple equal-length tensors (or ndarrays). Mirrors
    torch.utils.data.TensorDataset.
    """
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        n = len(np.asarray(tensors[0]))
        for t in tensors[1:]:
            if len(np.asarray(t)) != n:
                raise ValueError("TensorDataset: all tensors must have the same first-dim size")
        self.tensors = tuple(np.asarray(t) for t in tensors)
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)
