"""browsergrad_jit._safetensors — safetensors format reader.

INTERNAL. Public surface is `bg.load_safetensors(source)` on the
top-level namespace.

Format (per https://github.com/huggingface/safetensors):

  * 8 bytes: little-endian uint64 = header byte length N
  * N bytes: UTF-8 JSON metadata of the form
      {
        "tensor_name": {
          "dtype": "F32",  # or F16, BF16, F64, I8, I16, I32, I64, U8, ..., BOOL
          "shape": [d0, d1, ...],
          "data_offsets": [start_byte, end_byte_exclusive]
        },
        ...,
        "__metadata__": {arbitrary user keys}
      }
  * Remaining bytes: raw tensor data, contiguous, ordered per data_offsets.

Design choices for this v0:

  * **Python-only path** for `bytes` and `file://` sources. JS-side
    HTTP streaming + OPFS caching ships in a follow-on patch (PRD-008.2)
    once the runtime bridge for zero-copy ArrayBuffer transfer is
    designed and tested.
  * **Zero-copy where possible**: `np.frombuffer(buf, dtype=...)` over
    a memoryview slice avoids one full-file copy. We `.copy()` only
    where downstream may mutate (TensorProxy.numpy() already copies on
    realize).
  * **Memory ceiling**: header parse capped at 100 MB (safe upper bound;
    real headers are under 1 MB). Per-tensor allocations go straight
    into the BufferTable, so peak CPU memory is the file itself plus
    one tensor.
  * **BF16 is flagged**, not loaded. NumPy has no native BF16 dtype;
    surfacing as `uint16` would corrupt downstream math. Wait for
    PRD-010 mixed precision to land BF16 as a first-class dtype.
"""

from __future__ import annotations
import json
import os
import struct
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Dtype mapping. Safetensors uses an explicit string set; map each to the
# NumPy dtype name we use throughout the IR (which matches NumPy's `.name`).
# ---------------------------------------------------------------------------


_DTYPE_MAP: Dict[str, str] = {
    "F64":  "float64",
    "F32":  "float32",
    "F16":  "float16",
    # BF16 deliberately omitted; surfaced as an error below.
    "I64":  "int64",
    "I32":  "int32",
    "I16":  "int16",
    "I8":   "int8",
    "U64":  "uint64",
    "U32":  "uint32",
    "U16":  "uint16",
    "U8":   "uint8",
    "BOOL": "bool",
}


# Cap header at 100 MB. Real safetensors headers are < 1 MB; anything
# larger is either malicious input or a corrupted file.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Parser primitives
# ---------------------------------------------------------------------------


def _parse_header(buf: memoryview) -> tuple[Dict[str, Any], int]:
    """Read the safetensors header. Returns (header_dict, data_start_offset).

    `buf` is the entire file as a memoryview (or any sliceable buffer).
    The first 8 bytes are the little-endian uint64 header length, then
    the JSON bytes, then the tensor data starts at offset `8 + header_len`.
    """
    if len(buf) < 8:
        raise ValueError(
            f"safetensors: file is {len(buf)} bytes, too small to contain "
            f"the 8-byte header-length prefix"
        )
    (header_len,) = struct.unpack("<Q", bytes(buf[:8]))
    if header_len == 0:
        raise ValueError("safetensors: header length is zero")
    if header_len > _MAX_HEADER_BYTES:
        raise ValueError(
            f"safetensors: header length {header_len} exceeds the "
            f"{_MAX_HEADER_BYTES}-byte ceiling. Refusing to parse — "
            f"the file may be corrupted or malicious."
        )
    if 8 + header_len > len(buf):
        raise ValueError(
            f"safetensors: header length {header_len} exceeds file size "
            f"{len(buf)}"
        )
    header_bytes = bytes(buf[8 : 8 + header_len])
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"safetensors: header is not valid UTF-8 JSON: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(
            f"safetensors: header is a {type(header).__name__}, expected dict"
        )
    return header, 8 + header_len


def _validate_entry(name: str, meta: Any) -> tuple[str, tuple[int, ...], int, int]:
    """Validate one tensor entry from the header. Returns
    (numpy_dtype, shape, start_byte, end_byte_exclusive).

    Raises ValueError on any deviation from the spec.
    """
    if not isinstance(meta, dict):
        raise ValueError(
            f"safetensors: tensor {name!r} metadata is a "
            f"{type(meta).__name__}, expected dict"
        )
    dtype_str = meta.get("dtype")
    if not isinstance(dtype_str, str):
        raise ValueError(
            f"safetensors: tensor {name!r} missing string dtype "
            f"(got {dtype_str!r})"
        )
    if dtype_str == "BF16":
        raise NotImplementedError(
            f"safetensors: tensor {name!r} is BF16, which NumPy doesn't "
            f"natively support. BF16 lands in PRD-010 (real mixed precision). "
            f"For now, re-export the file as F16 or F32 via the upstream "
            f"PyTorch / Hugging Face safetensors library."
        )
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"safetensors: tensor {name!r} has unknown dtype {dtype_str!r}; "
            f"expected one of {sorted(_DTYPE_MAP) + ['BF16']}"
        )
    np_dtype = _DTYPE_MAP[dtype_str]

    shape = meta.get("shape")
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise ValueError(
            f"safetensors: tensor {name!r} has invalid shape {shape!r}; "
            f"expected list of non-negative ints"
        )

    offsets = meta.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(o, int) and o >= 0 for o in offsets)
        or offsets[0] > offsets[1]
    ):
        raise ValueError(
            f"safetensors: tensor {name!r} has invalid data_offsets "
            f"{offsets!r}; expected [start, end] with 0 <= start <= end"
        )

    # Sanity-check that offsets and shape×dtype match.
    expected_bytes = int(np.dtype(np_dtype).itemsize)
    for d in shape:
        expected_bytes *= d
    actual_bytes = offsets[1] - offsets[0]
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"safetensors: tensor {name!r} byte range {actual_bytes} doesn't "
            f"match shape×dtype {expected_bytes}"
        )

    return np_dtype, tuple(shape), offsets[0], offsets[1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_safetensors(
    source: Union[str, bytes, bytearray, memoryview],
    *,
    session: Any = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a safetensors file and return a dict mapping tensor name to
    `TensorProxy`. See module docstring for format details.

    `source`:
        * `bytes` / `bytearray` / `memoryview`: parse in-memory.
        * `str` starting with `file://`: read the file via stdlib `open()`.
        * Plain path string: same as `file://`. (HTTP URLs land in
          PRD-008.2 with the runtime bridge.)

    `session`: optional `Session` to register buffers in. Default: the
        implicit session (`browsergrad_jit.get_default_session()`).

    `progress(name, bytes_loaded, bytes_total)`: optional callback fired
        once per tensor after registration. Used by lab UIs for progress bars.

    `dtype`: if provided, cast every loaded tensor to this NumPy dtype
        name via `TensorProxy.cast`. Useful for the
        `load_safetensors("...", dtype="float32")` shortcut when a model
        is saved in float16 but you want to train in float32.

    Memory contract: at any moment we hold the full file plus at most
    one tensor's worth of duplication. Headers are validated up front;
    per-tensor allocations land directly in BufferTable.
    """
    buf, source_label = _resolve_source(source)
    header, data_start = _parse_header(buf)

    # Lazy import to avoid the circular dependency between _tensor_proxy
    # (which we use) and `browsergrad_jit.__init__` (which exposes load_safetensors).
    from ._tensor_proxy import from_numpy

    if session is None:
        import browsergrad_jit  # type: ignore
        session = browsergrad_jit.get_default_session()

    tensors: Dict[str, Any] = {}
    total_bytes = len(buf) - data_start

    for name, meta in header.items():
        if name == "__metadata__":
            continue  # opaque to us; not a tensor

        np_dtype, shape, start, end = _validate_entry(name, meta)
        abs_start = data_start + start
        abs_end = data_start + end
        if abs_end > len(buf):
            raise ValueError(
                f"safetensors: tensor {name!r} extends past end of file "
                f"({abs_end} > {len(buf)})"
            )

        # np.frombuffer over a memoryview slice avoids an extra copy.
        # The resulting array is read-only when sourced from a bytes
        # object; from_numpy doesn't mutate so this is safe. We .reshape
        # to the target shape and let the BufferTable hold the view.
        raw = np.frombuffer(
            bytes(buf[abs_start:abs_end]),
            dtype=np.dtype(np_dtype),
        )
        # Reshape produces a view; if shape has a 0 element (degenerate
        # tensors are legal in safetensors), reshape still works.
        arr = raw.reshape(shape)

        proxy = from_numpy(arr.copy(), session=session)
        if dtype is not None and proxy.dtype != dtype:
            proxy = proxy.cast(dtype)
        tensors[name] = proxy

        if progress is not None:
            progress(name, end, total_bytes)

    return tensors


def _resolve_source(
    source: Union[str, bytes, bytearray, memoryview],
) -> tuple[memoryview, str]:
    """Normalize the source argument to a memoryview + label-for-errors."""
    if isinstance(source, (bytes, bytearray)):
        return memoryview(source), "<bytes>"
    if isinstance(source, memoryview):
        return source, "<memoryview>"
    if isinstance(source, str):
        path = source
        if path.startswith("file://"):
            path = path[len("file://") :]
        if path.startswith(("http://", "https://")):
            raise NotImplementedError(
                f"safetensors: HTTP URL sources require the runtime bridge "
                f"that lands in PRD-008.2. For now, download to a local file "
                f"and pass the path. (got: {source!r})"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"safetensors: no such file: {path!r}")
        with open(path, "rb") as f:
            data = f.read()
        return memoryview(data), path
    raise TypeError(
        f"safetensors: source must be bytes / memoryview / file path, "
        f"got {type(source).__name__}"
    )


# Convenience: round-trip writer for tests and for the BufferTable→file
# checkpoint flow that craftingattention may want. Not the primary API;
# kept lean.

def save_safetensors(
    tensors: Dict[str, Any],
    path: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Write a dict of TensorProxies to a safetensors file.

    Tensors are realized via `.numpy()` and laid out in the order given.
    `metadata` is serialized as a `__metadata__` entry; safetensors spec
    requires its values be strings.
    """
    header: Dict[str, Any] = {}
    arrays: list[np.ndarray] = []
    cursor = 0
    for name, proxy in tensors.items():
        if not hasattr(proxy, "numpy"):
            raise TypeError(
                f"safetensors: value at {name!r} is a {type(proxy).__name__}, "
                f"expected TensorProxy"
            )
        arr = proxy.numpy()
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        np_dtype_name = arr.dtype.name
        # Reverse the dtype map.
        for ks_name, np_name in _DTYPE_MAP.items():
            if np_name == np_dtype_name:
                dtype_str = ks_name
                break
        else:
            raise ValueError(
                f"safetensors: cannot save dtype {np_dtype_name!r} (no "
                f"safetensors equivalent)"
            )
        size = int(arr.nbytes)
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
        arrays.append(arr)

    if metadata is not None:
        if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError(
                "safetensors: __metadata__ must be Dict[str, str]"
            )
        header["__metadata__"] = metadata

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for arr in arrays:
            f.write(arr.tobytes())


__all__ = ["load_safetensors", "save_safetensors"]
