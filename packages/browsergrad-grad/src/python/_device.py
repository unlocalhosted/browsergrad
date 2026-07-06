"""Optional KernelDevice bridge helpers for eager BrowserGrad.

The eager library remains NumPy-backed by default. Passing `device=` to the
small allowlist below routes forward data through a JS-side bridge, usually
created from `@unlocalhosted/browsergrad-kernels`.
"""

from __future__ import annotations
import numpy as np


def _as_float32_array(value, shape):
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(tuple(shape)).astype(np.float32, copy=False)


def matmul(device, a, b):
    if device is None:
        return None
    result = device.matmul(
        a.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(a.shape),
        b.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(b.shape),
    )
    return _as_float32_array(result, a.shape[:-1] + (b.shape[-1],))


def softmax(device, x):
    if device is None:
        return None
    result = device.softmax(
        x.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(x.shape),
    )
    return _as_float32_array(result, x.shape)


def layernorm(device, x, gamma, beta, eps):
    if device is None:
        return None
    gamma_data = None if gamma is None else gamma.astype(np.float32, copy=False).reshape(-1).tolist()
    beta_data = None if beta is None else beta.astype(np.float32, copy=False).reshape(-1).tolist()
    result = device.layernorm(
        x.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(x.shape),
        gamma_data,
        beta_data,
        float(eps),
    )
    return _as_float32_array(result, x.shape)


def attention(device, query, key, value):
    if device is None:
        return None
    result = device.attention(
        query.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(query.shape),
        key.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(key.shape),
        value.astype(np.float32, copy=False).reshape(-1).tolist(),
        tuple(value.shape),
    )
    return _as_float32_array(result, query.shape[:-1] + (value.shape[-1],))
