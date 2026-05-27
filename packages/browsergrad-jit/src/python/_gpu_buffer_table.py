"""browsergrad_jit._gpu_buffer_table — opaque GPU handle registry.

INTERNAL module. Mirrors BufferTable's shape but stores opaque handles
returned by the WebGPU bridge, NOT ndarrays. The bridge is the only
owner — Python never inspects what's inside a handle.

Lifecycle contract (per PRD-011.5 spike):
  * `register(buffer_id, handle)` — bind a BufferTable id to a GPU handle.
    Called on the first realize-webgpu walk after `bridge.upload()`
    materialises the seed buffer onto the GPU.
  * `get(buffer_id) -> handle | None` — look up; None means "not on GPU
    yet, caller must upload."
  * `release(buffer_id)` — drop the handle (calls `bridge.release(handle)`
    so the GPU side can free the GPUBuffer).
  * `release_all()` — used by session teardown.

The buffer_id space is shared with BufferTable on purpose: a LOAD UOp
wraps a BUFFER node carrying a buffer_id, and the WebGPU realizer needs
to translate that id into a GPU handle. Keying both tables on the same
id space means no extra correlation layer.

Why this is a separate file from BufferTable: BufferTable is the
NumPy realizer's source of truth and runs in every Pyodide session.
GpuBufferTable only exists when a WebGPU bridge is registered. Keeping
it separate means non-WebGPU users never pay the import cost and the
NumPy realizer can ignore it entirely.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ._errors import BufferTableError


class GpuBufferTable:
    """Per-session registry mapping BufferTable ids → opaque GPU handles."""

    __slots__ = ("_handles", "_bridge", "_total_uploaded_bytes", "_handles_alive")

    def __init__(self, bridge: Any) -> None:
        """`bridge` must implement the WebGpuBridge protocol (see _bridge.py).
        In production the bridge is a Pyodide JsProxy wrapping
        `createWebGpuRealizerBridge(device)` from browsergrad-kernels; in
        tests a Python-side mock implementing the same surface."""
        if bridge is None:
            raise BufferTableError(
                "GpuBufferTable requires a non-None bridge. Call "
                "bg.register_webgpu_bridge(bridge) before constructing one."
            )
        self._handles: Dict[str, Any] = {}
        self._bridge = bridge
        self._total_uploaded_bytes: int = 0
        # Diagnostic — counts handles that have ever been minted minus those
        # released. Drives the residency-proof tests (chained matmuls should
        # have at most N+1 alive after upload, where N is the number of
        # intermediate ops).
        self._handles_alive: int = 0

    @property
    def bridge(self) -> Any:
        return self._bridge

    def register(self, buffer_id: str, handle: Any) -> None:
        """Bind `buffer_id` → `handle`. Idempotent: re-registering the
        same id raises (catches an entire class of double-upload bugs
        where the realizer forgets to consult the table)."""
        if buffer_id in self._handles:
            raise BufferTableError(
                f"GpuBufferTable: buffer_id {buffer_id!r} is already "
                f"registered with a handle. The realizer should consult "
                f".get() before re-uploading."
            )
        self._handles[buffer_id] = handle
        self._handles_alive += 1

    def get(self, buffer_id: str) -> Optional[Any]:
        return self._handles.get(buffer_id)

    def has(self, buffer_id: str) -> bool:
        return buffer_id in self._handles

    def release(self, buffer_id: str) -> None:
        handle = self._handles.pop(buffer_id, None)
        if handle is None:
            return
        self._bridge.release(handle)
        self._handles_alive -= 1

    def release_all(self) -> None:
        for buffer_id in list(self._handles):
            self.release(buffer_id)

    def upload(self, buffer_id: str, data: bytes, shape: tuple, dtype: str) -> Any:
        """Convenience: upload via bridge and register in one step.
        Returns the handle. Idempotent; a second upload returns the
        already-registered handle without re-crossing the bridge."""
        existing = self._handles.get(buffer_id)
        if existing is not None:
            return existing
        handle = self._bridge.upload(data, shape, dtype)
        self._handles[buffer_id] = handle
        self._handles_alive += 1
        self._total_uploaded_bytes += len(data)
        return handle

    def stats(self) -> Dict[str, int]:
        """Snapshot for observability + tests. The residency proof asserts
        on `handles_alive` and `uploaded_bytes` after a chained-matmul run."""
        return {
            "handles_alive": self._handles_alive,
            "uploaded_bytes": self._total_uploaded_bytes,
            "registered_ids": len(self._handles),
        }


__all__ = ["GpuBufferTable"]
