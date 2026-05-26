"""browsergrad_jit._torch_compat — `install_torch_alias()` machinery.

INTERNAL module. The functions are re-exported from the top-level
`browsergrad_jit` namespace so users can:

    import browsergrad_jit
    browsergrad_jit.install_torch_alias()
    import torch          # → maps to browsergrad_jit

Owner-token protocol (per PRD-005 critique P1-2):

  Both browsergrad_jit and browsergrad_grad can call install_torch_alias().
  Each tags the resulting `sys.modules["torch"]` with `__bg_owner__` set
  to its package name. If a second package tries to install while the
  first still owns the alias, `TorchAliasConflict` is raised explaining
  the situation — preventing the "identity invariant" check in either
  package's torch_compat shim from firing with a confusing
  `AssertionError` deep in user code.

  `uninstall_torch_alias()` cleanly releases the owner tag if the current
  process owns it; otherwise no-op.

  `force=True` overrides the conflict for testing; documented as such.
"""

from __future__ import annotations
import sys
import types

from ._errors import TorchAliasConflict


OWNER_NAME = "browsergrad_jit"


def install_torch_alias(*, force: bool = False) -> None:
    """Register browsergrad_jit as `torch` in sys.modules.

    After this call, `import torch` returns the browsergrad_jit module —
    plus `torch.nn`, `torch.optim`, `torch.nn.functional`. Existing user
    code written against PyTorch shapes runs unchanged for the supported
    op surface; ops not yet implemented in browsergrad_jit raise
    `JitNotImplementedError`.

    If sys.modules["torch"] is already owned by another package (e.g.
    browsergrad_grad), raises TorchAliasConflict unless force=True.
    """
    existing = sys.modules.get("torch")
    if existing is not None:
        owner = getattr(existing, "__bg_owner__", None)
        if owner == OWNER_NAME:
            return  # idempotent
        if owner is not None and not force:
            raise TorchAliasConflict(
                f"sys.modules['torch'] is owned by {owner!r}. Call "
                f"{owner}.uninstall_torch_alias() first, or pass force=True "
                f"to override (tests do this between scenarios)."
            )
        if owner is None and not force:
            # Real PyTorch (or some other unrelated module) is in sys.modules.
            # Refuse — overwriting real PyTorch would silently break code.
            raise TorchAliasConflict(
                "sys.modules['torch'] already exists and is not owned by "
                "a browsergrad package. Refusing to shadow it; pass force=True "
                "if you really want to."
            )

    import browsergrad_jit
    torch_mod = types.ModuleType("torch")
    torch_mod.__bg_owner__ = OWNER_NAME

    # Re-export every public name from browsergrad_jit onto the torch alias.
    for name in browsergrad_jit.__all__:
        setattr(torch_mod, name, getattr(browsergrad_jit, name))

    # Sub-namespaces also need their own sys.modules entries so
    # `import torch.nn`, `import torch.nn.functional`, `import torch.optim`
    # resolve correctly.
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = browsergrad_jit.nn
    sys.modules["torch.nn.functional"] = browsergrad_jit._functional
    sys.modules["torch.optim"] = browsergrad_jit.optim


def uninstall_torch_alias() -> None:
    """Remove the torch alias if this package owns it.

    No-op if sys.modules["torch"] doesn't exist or is owned by another
    package — the call is always safe to make."""
    existing = sys.modules.get("torch")
    if existing is None:
        return
    if getattr(existing, "__bg_owner__", None) != OWNER_NAME:
        return
    sys.modules.pop("torch", None)
    sys.modules.pop("torch.nn", None)
    sys.modules.pop("torch.nn.functional", None)
    sys.modules.pop("torch.optim", None)


__all__ = ["install_torch_alias", "uninstall_torch_alias"]
