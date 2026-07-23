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

    __slots__ = (
        "_buffers",
        "_metadata",
        "_session_token",
        "_effect_streams",
        "_effect_stream_tokens",
    )

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[np.ndarray]] = {}
        self._metadata: dict[str, Tuple[Tuple[int, ...], str]] = {}
        # A token tagging this table — used to detect accidental cross-session
        # UOp reuse. Two BufferTables in the same process have different
        # tokens; if a UOp built against session A is realized in session B,
        # we can refuse rather than producing a wrong-array result.
        self._session_token: str = uuid.uuid4().hex[:12]
        # Stateful typed operations use one ordered effect stream per exact
        # (kind, target-buffer tuple). Only the reserved/applied sequence
        # watermarks are retained, so a long-running training loop consumes
        # O(stateful modules) memory rather than O(forward calls).
        self._effect_streams: dict[
            str, Tuple[str, Tuple[str, ...], int, int]
        ] = {}
        self._effect_stream_tokens: dict[
            Tuple[str, Tuple[str, ...]], str
        ] = {}

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

    def reserve_effect(
        self,
        kind: str,
        target_buffer_ids: Tuple[str, ...],
    ) -> str:
        """Mint a session-owned, initially-unapplied effect identity.

        ``kind`` is deliberately closed to a small canonical alphabet so an
        effect id is safe to retain in IR, diagnostics, and serialized test
        evidence without permitting caller-controlled formatting payloads.
        """
        if (
            type(kind) is not str
            or not kind
            or len(kind) > 64
            or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in kind)
        ):
            raise BufferTableError(
                "effect kind must be 1..64 lowercase ASCII letters, digits, or '-'"
            )
        if (
            type(target_buffer_ids) is not tuple
            or not target_buffer_ids
            or len(target_buffer_ids) > 16
            or len(set(target_buffer_ids)) != len(target_buffer_ids)
        ):
            raise BufferTableError(
                "effect targets must be a tuple of 1..16 unique buffer ids"
            )
        for index, buffer_id in enumerate(target_buffer_ids):
            if type(buffer_id) is not str:
                raise BufferTableError(
                    f"effect target {index} must be a buffer id string"
                )
            # Resolve now so a reservation can never carry an unknown,
            # cross-session, or unmaterialized target.
            self.get(buffer_id)
        stream_key = (kind, target_buffer_ids)
        token = self._effect_stream_tokens.get(stream_key)
        if token is None:
            token = uuid.uuid4().hex
            self._effect_stream_tokens[stream_key] = token
            self._effect_streams[token] = (
                kind,
                target_buffer_ids,
                0,
                0,
            )
        stream_kind, stream_targets, reserved, applied = (
            self._effect_streams[token]
        )
        reserved += 1
        self._effect_streams[token] = (
            stream_kind,
            stream_targets,
            reserved,
            applied,
        )
        return f"{self._session_token}:effect:{token}:{reserved}"

    def _validate_effect(
        self,
        effect_id: str,
        expected_kind: str,
    ) -> Tuple[bool, Tuple[str, ...], str, int]:
        prefix = self._session_token + ":effect:"
        if type(effect_id) is not str or not effect_id.startswith(prefix):
            raise BufferTableError(
                f"effect_id {effect_id!r} does not belong to session "
                f"{self._session_token!r}"
            )
        payload = effect_id[len(prefix):]
        token, separator, sequence_text = payload.partition(":")
        if (
            separator != ":"
            or len(token) != 32
            or any(ch not in "0123456789abcdef" for ch in token)
            or not sequence_text
            or len(sequence_text) > 20
            or not sequence_text.isascii()
            or not sequence_text.isdigit()
            or ":" in sequence_text
        ):
            raise BufferTableError(
                f"effect_id {effect_id!r} is malformed"
            )
        sequence = int(sequence_text)
        record = self._effect_streams.get(token)
        if record is None:
            raise BufferTableError(
                f"effect_id {effect_id!r} was not reserved by this session"
            )
        kind, targets, reserved, applied = record
        if kind != expected_kind:
            raise BufferTableError(
                f"effect_id {effect_id!r} has kind {kind!r}, expected "
                f"{expected_kind!r}"
            )
        if sequence < 1 or sequence > reserved:
            raise BufferTableError(
                f"effect_id {effect_id!r} has unreserved sequence {sequence}"
            )
        return sequence <= applied, targets, token, sequence

    def effect_applied(self, effect_id: str, expected_kind: str) -> bool:
        """Return whether a reserved session effect has committed."""
        applied, _, _, _ = self._validate_effect(effect_id, expected_kind)
        return applied

    def apply_updates_once(
        self,
        effect_id: str,
        expected_kind: str,
        updates: Tuple[Tuple[str, np.ndarray], ...],
    ) -> bool:
        """Atomically validate and apply in-place buffer updates once.

        All target identities, shapes, and dtypes are checked before any
        mutation. Updates preserve the registered ndarray objects so public
        module-buffer identity and state-dict references remain stable.
        Returns ``True`` only for the first successful commit.
        """
        applied, bound_targets, stream_token, sequence = self._validate_effect(
            effect_id, expected_kind
        )
        if type(updates) is not tuple or not updates:
            raise BufferTableError("effect updates must be a non-empty tuple")

        validated: list[Tuple[np.ndarray, np.ndarray]] = []
        seen: set[str] = set()
        for index, update in enumerate(updates):
            if type(update) is not tuple or len(update) != 2:
                raise BufferTableError(
                    f"effect updates[{index}] must be (buffer_id, ndarray)"
                )
            buffer_id, new_value = update
            if type(buffer_id) is not str or buffer_id in seen:
                raise BufferTableError(
                    f"effect updates[{index}] has an invalid or duplicate buffer id"
                )
            seen.add(buffer_id)
            if not isinstance(new_value, np.ndarray):
                raise BufferTableError(
                    f"effect updates[{index}] value must be np.ndarray"
                )
            target = self.get(buffer_id)
            expected_shape, expected_dtype = self._metadata[buffer_id]
            if tuple(new_value.shape) != expected_shape:
                raise BufferTableError(
                    f"shape mismatch on effect update of {buffer_id!r}: "
                    f"existing {expected_shape}, new {new_value.shape}"
                )
            if new_value.dtype.name != expected_dtype:
                raise BufferTableError(
                    f"dtype mismatch on effect update of {buffer_id!r}: "
                    f"existing {expected_dtype}, new {new_value.dtype}"
                )
            validated.append((target, new_value))

        if tuple(seen_id for seen_id, _ in updates) != bound_targets:
            raise BufferTableError(
                f"effect_id {effect_id!r} is bound to targets "
                f"{bound_targets!r}, not {tuple(seen_id for seen_id, _ in updates)!r}"
            )
        if applied:
            return False
        stream_kind, stream_targets, reserved, applied_sequence = (
            self._effect_streams[stream_token]
        )
        if sequence != applied_sequence + 1:
            raise BufferTableError(
                f"effect_id {effect_id!r} is out of order: next sequence is "
                f"{applied_sequence + 1}"
            )
        for target, new_value in validated:
            target[...] = new_value
        self._effect_streams[stream_token] = (
            stream_kind,
            stream_targets,
            reserved,
            sequence,
        )
        return True

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
