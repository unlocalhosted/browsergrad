"""Pile B — possible-but-limited shims.

Ships with explicit caveats: autocast is a no-op (no real fp16 path in WASM),
linalg wraps numpy.linalg (forward-only, no autograd), and Module.to accepts
CPU identity while rejecting unavailable placement or dtype conversion.

Depends on Pile A having registered _bg.nn.Module already — the monkey-patch
on line `_bg.nn.Module.to = ...` reaches into the live class. The orchestrator
asserts `torch_mod.nn.Module is _bg.nn.Module` before calling us, so this
patch propagates to the torch namespace via Python's reference semantics.
"""

import contextlib as _ctxlib
import numpy as _np


def install_limited(torch_mod, _bg, _types):
    # torch.amp.autocast — no-op context manager. We have no real fp16 path
    # in WASM/Pyodide; calling autocast() shouldn't break code that uses it.
    torch_amp = _types.ModuleType("torch.amp")
    @_ctxlib.contextmanager
    def _autocast(*args, **kwargs):
        yield
    torch_amp.autocast = _autocast
    torch_mod.amp = torch_amp

    # torch.linalg — wrap numpy.linalg, return Tensors.
    torch_linalg = _types.ModuleType("torch.linalg")
    def _linalg_norm(t, ord=None, dim=None, keepdim=False):
        arr = _np.asarray(t)
        out = _np.linalg.norm(arr, ord=ord, axis=dim, keepdims=keepdim)
        return _bg.Tensor(out.astype(_np.float32))
    def _linalg_inv(t):
        return _bg.Tensor(_np.linalg.inv(_np.asarray(t)).astype(_np.float32))
    def _linalg_det(t):
        return _bg.Tensor(_np.linalg.det(_np.asarray(t)).astype(_np.float32))
    def _linalg_svd(t):
        u, s, vh = _np.linalg.svd(_np.asarray(t))
        return _bg.Tensor(u.astype(_np.float32)), _bg.Tensor(s.astype(_np.float32)), _bg.Tensor(vh.astype(_np.float32))
    def _linalg_eigh(t):
        w, v = _np.linalg.eigh(_np.asarray(t))
        return _bg.Tensor(w.astype(_np.float32)), _bg.Tensor(v.astype(_np.float32))
    def _linalg_solve(a, b):
        out = _np.linalg.solve(_np.asarray(a), _np.asarray(b))
        return _bg.Tensor(out.astype(_np.float32))
    def _linalg_pinv(t):
        return _bg.Tensor(_np.linalg.pinv(_np.asarray(t)).astype(_np.float32))
    torch_linalg.norm = _linalg_norm
    torch_linalg.inv = _linalg_inv
    torch_linalg.det = _linalg_det
    torch_linalg.svd = _linalg_svd
    torch_linalg.eigh = _linalg_eigh
    torch_linalg.solve = _linalg_solve
    torch_linalg.pinv = _linalg_pinv
    torch_mod.linalg = torch_linalg

    # CPU-only module placement shim. It must not imply unavailable device
    # storage or silently ignore parameter dtype conversion.
    # CRITICAL: this patches the LIVE _bg.nn.Module class. It only propagates
    # to torch.nn.Module because Pile A's shallow copy preserves class
    # identity (see the orchestrator's assertion). If anyone deep-copies
    # _bg.nn into torch_nn, remove this patch and add Module.to to nn.py.
    def _module_to_shim(self, *args, **kwargs):
        unsupported_kwargs = sorted(set(kwargs) - {"device", "dtype"})
        if unsupported_kwargs:
            raise TypeError(
                "nn.Module.to only supports device= and dtype=; unsupported "
                f"keyword(s): {', '.join(unsupported_kwargs)}"
            )
        if len(args) > 1:
            raise TypeError(
                f"nn.Module.to accepts at most one positional argument, got {len(args)}"
            )
        dtype_provided = "dtype" in kwargs
        device_provided = "device" in kwargs
        dtype_spec = kwargs.get("dtype")
        device_spec = kwargs.get("device")
        if args:
            first = args[0]
            if (
                isinstance(first, str)
                and first.split(":", 1)[0] in ("cpu", "cuda", "mps", "xpu", "meta")
            ):
                if device_provided:
                    raise TypeError("nn.Module.to received device more than once")
                device_spec = first
            else:
                if dtype_provided:
                    raise TypeError("nn.Module.to received dtype more than once")
                dtype_spec = first

        if device_spec is not None:
            if not isinstance(device_spec, str):
                raise TypeError(
                    "nn.Module.to device must be the string 'cpu'; "
                    f"got {type(device_spec).__name__}"
                )
            if device_spec != "cpu":
                raise NotImplementedError(
                    f"nn.Module.to({device_spec!r}) is unavailable: eager Grad "
                    "parameters are CPU/Pyodide-backed and no transfer occurred"
                )
        if dtype_spec is not None:
            raise NotImplementedError(
                f"nn.Module.to(dtype={dtype_spec!r}) is unavailable: module "
                "parameter conversion is not implemented and no dtype changed"
            )
        return self
    _bg.nn.Module.to = _module_to_shim
