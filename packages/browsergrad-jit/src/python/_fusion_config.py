"""browsergrad_jit._fusion_config — runtime control for the fusion pass.

INTERNAL. The public surface lives on the top-level package as
`bg.jit.use_fusion(bool)` / `bg.jit.debug_fused_kernels()` /
`bg.jit.debug_unfused_reasons()`.

Two ways to disable fusion:
  * `use_fusion(False)` — process-level Python switch. Useful for
    correctness debugging from inside a Pyodide session.
  * `BG_DISABLE_FUSION=1` env var — wins over the Python switch.
    Useful for CI runs that test the unfused path globally.

By default fusion is ON. The realizer asks `is_enabled()` once per
realization call; the cost is a function call plus an env-var lookup the
first time. Cheap enough not to bother memoizing.
"""

from __future__ import annotations
import os


_ENABLED: bool = True
_ENV_CACHED: bool | None = None


def _env_disables() -> bool:
    """Cache the env-var lookup. If set at process start to "1" / "true",
    fusion is forced off regardless of the Python switch."""
    global _ENV_CACHED
    if _ENV_CACHED is None:
        val = os.environ.get("BG_DISABLE_FUSION", "").lower()
        _ENV_CACHED = val in ("1", "true", "yes", "on")
    return _ENV_CACHED


def is_enabled() -> bool:
    return _ENABLED and not _env_disables()


def use_fusion(enabled: bool) -> None:
    """Toggle fusion globally. The default is True.

    `BG_DISABLE_FUSION=1` overrides this — if the env var disables
    fusion, `use_fusion(True)` is a silent no-op. This matches the
    Unix convention that the env var is the user's last-resort
    escape hatch."""
    global _ENABLED
    _ENABLED = bool(enabled)


__all__ = ["use_fusion", "is_enabled"]
