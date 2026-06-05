"""browsergrad_grad.torch_compat — install a torch-namespace shim.

The shim is split across three private modules reflecting the Pile A/B/C
taxonomy from ARCHITECTURE.md / PROGRESS.md:

* `_torch_compat_real` — Pile A: real op aliases.
* `_torch_compat_limited` — Pile B: shims with explicit caveats.
* `_torch_compat_impossible` — Pile C: loud NotImplementedError stubs.

`install_torch_alias` orchestrates: builds an empty torch module, calls each
pile's install function in order, asserts the cross-pile identity invariant,
and registers everything on `sys.modules`.
"""

import sys as _bg_torch_sys
import types as _bg_torch_types

from ._torch_compat_real import install_real
from ._torch_compat_limited import install_limited
from ._torch_compat_impossible import install_impossible


def install_torch_alias():
    """Register torch / torch.nn / torch.nn.functional / torch.optim in
    sys.modules so PyTorch user code runs against browsergrad_grad.

    Calling repeatedly is safe — the alias is re-installed each time, so
    if a user has reset sys.modules between calls it'll rebuild.
    """
    import browsergrad_grad as _bg

    torch_mod = _bg_torch_types.ModuleType("torch")
    torch_mod.__doc__ = "browsergrad_grad shim under the torch namespace"

    install_real(torch_mod, _bg, _bg_torch_types)

    # Identity invariant: Pile B patches `_bg.nn.Module.to` in place, and we
    # need that patch to be visible via `torch.nn.Module.to`. That only works
    # because Pile A's `setattr(torch_nn, name, getattr(_bg.nn, name))` copies
    # the CLASS REFERENCE, not a fresh copy, so torch_nn.Module IS _bg.nn.Module.
    # If anyone ever switches Pile A to a deep copy, this assertion fires and
    # Pile B's _module_to_shim patch must move into nn.py as a real method.
    assert torch_mod.nn.Module is _bg.nn.Module, (
        "torch_compat invariant violated: torch_mod.nn.Module must be identical "
        "to _bg.nn.Module (shallow class reference). If you changed install_real "
        "to deep-copy _bg.nn, move Module.to into nn.py instead of monkey-patching."
    )

    install_limited(torch_mod, _bg, _bg_torch_types)
    install_impossible(torch_mod, _bg_torch_types)

    # Register sys.modules entries — read sub-modules off torch_mod so we
    # don't have to know whether they were added by Pile A or Pile B/C.
    _bg_torch_sys.modules["torch"] = torch_mod
    _bg_torch_sys.modules["torch.nn"] = torch_mod.nn
    _bg_torch_sys.modules["torch.nn.functional"] = torch_mod.nn.functional
    _bg_torch_sys.modules["torch.nn.utils"] = torch_mod.nn.utils
    _bg_torch_sys.modules["torch.optim"] = torch_mod.optim
    _bg_torch_sys.modules["torch.utils"] = torch_mod.utils
    _bg_torch_sys.modules["torch.utils.data"] = torch_mod.utils.data
    _bg_torch_sys.modules["torch.amp"] = torch_mod.amp
    _bg_torch_sys.modules["torch.linalg"] = torch_mod.linalg
    _bg_torch_sys.modules["torch.fx"] = torch_mod.fx
    _bg_torch_sys.modules["torch.jit"] = torch_mod.jit
    _bg_torch_sys.modules["torch.cuda"] = torch_mod.cuda
    _bg_torch_sys.modules["torch.distributed"] = torch_mod.distributed
    _bg_torch_sys.modules["torch.onnx"] = torch_mod.onnx
    _bg_torch_sys.modules["torch.quantization"] = torch_mod.quantization

    return torch_mod
