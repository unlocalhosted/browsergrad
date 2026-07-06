"""browsergrad_jit._buffer_table — per-session storage for realized arrays.

INTERNAL. Lifecycle:

  * Each Pyodide session (in the lab runtime sense) creates exactly one
    `BufferTable` and threads it through every realization call. The table
    is the single source of truth for "what's the concrete NumPy array
    behind buffer_id X."
  * NEVER a module global. The PRD-005 critique called out the original
    design's module-level dict as a P0 ship-blocker: two `nn.Module`s in
    the same Pyodide worker would collide on buffer ids and silently
    corrupt each other. The per-session design forecloses that bug class.
  * Cross-session sharing is undefined. If you find yourself wanting it,
    serialize via state_dict / safetensors — not by reaching into the
    table directly.

The table is intentionally minimal — it's a dict plus a few invariants.
The expensive logic (realization, dispatch) lives elsewhere; this file
exists only to enforce the lifecycle rules and produce good error
messages when callers break them.
"""

from __future__ import annotations
from typing import Iterator, Optional, Tuple
import uuid

import numpy as np

from ._errors import BufferTableError


class BufferTable:
    """Per-session map from `buffer_id` → np.ndarray.

    Instances are NOT thread-safe (Pyodide is single-threaded; if that ever
    changes, this becomes the synchronization point). They ARE safe to
    share across nested `eager_context()` regions because all reads/writes
    go through this object.

    Buffer ids are short hex strings minted from uuid4. Callers may pass
    a `name` to `new_buffer` for debuggability; if omitted, the id is
    purely opaque.
    """

    __slots__ = ("_buffers", "_metadata", "_session_token")

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[np.ndarray]] = {}
        self._metadata: dict[str, Tuple[Tuple[int, ...], str]] = {}
        # A token tagging this table — used to detect accidental cross-session
        # UOp reuse. Two BufferTables in the same process have different
        # tokens; if a UOp built against session A is realized in session B,
        # we can refuse rather than producing a wrong-array result.
        self._session_token: str = uuid.uuid4().hex[:12]

    @property
    def session_token(self) -> str:
        """Opaque identifier for this BufferTable. Used by realization to
        verify a UOp was built for this session. Not stable across processes."""
        return self._session_token

    def new_buffer(self, array: np.ndarray, name: Optional[str] = None) -> str:
        """Register `array` and return its `buffer_id`.

        If `name` is given it's used as the id (after a session-token
        prefix to keep ids unique-per-session); otherwise a uuid4-derived
        id is minted. Raises `BufferTableError` on duplicate registration.

        The array is stored by reference. Callers must not mutate the
        array in place after registration — the realizer assumes buffer
        contents are stable for the lifetime of any UOp graph that
        references them. (Optimizer step is the legitimate exception; it
        goes through the STORE opcode, not direct mutation.)
        """
        if not isinstance(array, np.ndarray):
            raise BufferTableError(
                f"new_buffer expected np.ndarray, got {type(array).__name__}"
            )
        if name is not None:
            buffer_id = f"{self._session_token}:{name}"
        else:
            buffer_id = f"{self._session_token}:{uuid.uuid4().hex[:10]}"
        if buffer_id in self._buffers:
            raise BufferTableError(
                f"buffer_id {buffer_id!r} already registered in this session"
            )
        self._buffers[buffer_id] = array
        self._metadata[buffer_id] = (tuple(array.shape), array.dtype.name)
        return buffer_id

    def new_unmaterialized_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        name: Optional[str] = None,
    ) -> str:
        """Reserve a buffer id whose bytes currently live outside NumPy.

        Used by explicit WebGPU-resident tensor-plan realization. CPU
        realization refuses until the owner materializes bytes back through
        `update()`.
        """
        if name is not None:
            buffer_id = f"{self._session_token}:{name}"
        else:
            buffer_id = f"{self._session_token}:{uuid.uuid4().hex[:10]}"
        if buffer_id in self._buffers:
            raise BufferTableError(
                f"buffer_id {buffer_id!r} already registered in this session"
            )
        self._buffers[buffer_id] = None
        self._metadata[buffer_id] = (tuple(shape), np.dtype(dtype).name)
        return buffer_id

    def get(self, buffer_id: str) -> np.ndarray:
        """Look up the array for `buffer_id`. Raises if unknown.

        Refuses ids from other sessions — the prefix check catches the
        cross-session contamination bug PRD-005 critique called out.
        """
        if not buffer_id.startswith(self._session_token + ":"):
            owner = buffer_id.split(":", 1)[0] if ":" in buffer_id else "<no-session>"
            raise BufferTableError(
                f"buffer_id {buffer_id!r} belongs to session {owner!r}, "
                f"but this is session {self._session_token!r}. "
                f"UOps are not portable across sessions; rebuild the graph "
                f"in the current session before realizing."
            )
        if buffer_id not in self._buffers:
            raise BufferTableError(
                f"unknown buffer_id {buffer_id!r} in this session. "
                f"Was it cleared? (Currently {len(self._buffers)} buffers registered.)"
            )
        arr = self._buffers[buffer_id]
        if arr is None:
            shape, dtype = self._metadata[buffer_id]
            raise BufferTableError(
                f"buffer_id {buffer_id!r} is GPU-resident and not materialized "
                f"in the CPU BufferTable yet (shape={shape}, dtype={dtype}). "
                f"Call .numpy(), .item(), or realize through a WebGPU path."
            )
        return arr

    def is_materialized(self, buffer_id: str) -> bool:
        return buffer_id in self._buffers and self._buffers[buffer_id] is not None

    def metadata(self, buffer_id: str) -> Tuple[Tuple[int, ...], str]:
        if buffer_id not in self._metadata:
            raise BufferTableError(f"unknown buffer_id {buffer_id!r} in this session")
        return self._metadata[buffer_id]

    def mark_unmaterialized(self, buffer_id: str) -> None:
        """Mark an existing buffer id as GPU-owned/CPU-unmaterialized."""
        if buffer_id not in self._buffers:
            raise BufferTableError(
                f"cannot mark unknown buffer_id {buffer_id!r} unmaterialized"
            )
        self._buffers[buffer_id] = None

    def update(self, buffer_id: str, array: np.ndarray) -> None:
        """Replace the array for `buffer_id`. Used by STORE on optimizer
        steps and by `.grad` accumulation.

        Refuses if shape/dtype changed — that would indicate a graph bug
        upstream, not a legitimate mutation."""
        if buffer_id not in self._buffers:
            raise BufferTableError(
                f"cannot update unknown buffer_id {buffer_id!r}"
            )
        expected_shape, expected_dtype = self._metadata[buffer_id]
        if expected_shape != tuple(array.shape):
            raise BufferTableError(
                f"shape mismatch on update of {buffer_id!r}: "
                f"existing {expected_shape}, new {array.shape}"
            )
        if expected_dtype != array.dtype.name:
            raise BufferTableError(
                f"dtype mismatch on update of {buffer_id!r}: "
                f"existing {expected_dtype}, new {array.dtype}"
            )
        self._buffers[buffer_id] = array

    def evict(self, buffer_id: str) -> None:
        """Drop a buffer. Intended for tests and for gradient-checkpoint
        eviction (PRD-009). User code should not call this directly."""
        if buffer_id not in self._buffers:
            raise BufferTableError(
                f"cannot evict unknown buffer_id {buffer_id!r}"
            )
        del self._buffers[buffer_id]
        self._metadata.pop(buffer_id, None)

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._buffers)

    def __repr__(self) -> str:
        return (
            f"BufferTable(session={self._session_token!r}, "
            f"n_buffers={len(self._buffers)})"
        )


__all__ = ["BufferTable"]
