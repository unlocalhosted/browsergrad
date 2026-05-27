"""browsergrad_jit._custom_kernel — user-supplied WGSL kernels (PRD-015).

INTERNAL. Public surface: `@bg.custom_kernel(wgsl=..., name=..., ...)`.

Design (per the DL/GPU review's cut):
  * Forward-only. No backward registration in v0. Calling .backward()
    on a tensor downstream of a user kernel raises NoBackwardError
    (OP_CUSTOM has no registered VJP rule — same as flash_attention).
  * The decorator produces a callable that builds an OP_CUSTOM UOp
    tagged "user" with the kernel hash. The WebGPU realizer dispatches
    via bridge.run_user_kernel().
  * No template engine. Users hand-craft WGSL with concrete shapes
    baked in. Shape specialization is a follow-on PRD.
  * On the NumPy realizer the user kernel raises with a clear pointer
    to bg.realize_webgpu — keeps the educational story honest.

Lifecycle:
  * Each decorator call registers a KernelSpec in a process-global
    registry keyed by SHA-256 of the WGSL source.
  * At realize time the WebGPU realizer pulls the spec from the
    registry and hands the bridge the WGSL + dispatch metadata.
  * The bridge's pipeline cache (via runner.ts) handles compile-once
    semantics — same WGSL → same cache entry → no recompile.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from ._ir import UOp, OP_CUSTOM
from ._errors import JitNotImplementedError


@dataclass(frozen=True)
class KernelSpec:
    """Registered user-WGSL kernel. Immutable — re-registering with a
    different shape but the same WGSL is fine; the registry only stores
    one spec per hash."""
    wgsl: str
    name: str
    workgroup_size: Tuple[int, int, int]
    num_inputs: int
    output_dtype: str
    hash: str


_REGISTRY: Dict[str, KernelSpec] = {}


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_registry() -> Dict[str, KernelSpec]:
    """Public introspection — the WebGPU bridge reads this to fetch WGSL
    by hash. Returns a live reference; do not mutate from user code."""
    return _REGISTRY


def custom_kernel(
    wgsl: str,
    name: str,
    workgroup_size: Tuple[int, int, int],
    output_shape_fn: Callable[..., Tuple[int, ...]],
    dispatch_shape_fn: Callable[..., Tuple[int, int, int]],
    num_inputs: int,
    output_dtype: str = "float32",
) -> Callable[..., Any]:
    """Register a user-supplied WGSL kernel and return a callable that
    builds an OP_CUSTOM UOp at runtime.

    Parameters
    ----------
    wgsl
        The WGSL source. Must declare bindings @group(0) @binding(0..num_inputs-1)
        as read-only-storage inputs, @group(0) @binding(num_inputs) as the
        read_write-storage output, and (optionally) @group(0) @binding(num_inputs+1)
        as a uniform param struct (v0 ships without param-uniform support;
        users bake constants into WGSL).
    name
        Human-readable kernel name. Prefixed with hash[:8] for cache keys
        so two kernels with the same name but different WGSL coexist.
    workgroup_size
        Triple matching @workgroup_size(...) in the WGSL. Mismatch → WebGPU
        validation error at pipeline creation.
    output_shape_fn
        `(*input_shapes) -> output_shape`. Pure Python. Called at trace time
        to populate the OP_CUSTOM UOp's shape.
    dispatch_shape_fn
        `(*input_shapes) -> (x, y, z)` dispatch count. The runner divides
        by workgroup_size internally.
    num_inputs
        Number of input tensors the kernel takes. Validated at call time.
    output_dtype
        Dtype of the output tensor; default float32.

    Returns
    -------
    A callable `kernel(*inputs) -> TensorProxy`. The callable constructs
    an OP_CUSTOM UOp tagged "user" with the kernel hash; realize via
    bg.realize_webgpu (NumPy realizer refuses).
    """
    if not isinstance(wgsl, str) or not wgsl.strip():
        raise ValueError("custom_kernel: wgsl must be a non-empty string")
    if num_inputs < 1 or num_inputs > 8:
        raise ValueError(
            f"custom_kernel: num_inputs={num_inputs} outside [1, 8] (the "
            f"WGSL bind-group layout caps inputs at 8 in v0)"
        )
    if output_dtype != "float32":
        raise NotImplementedError(
            f"custom_kernel: only float32 supported in v0 (got {output_dtype!r}). "
            f"f16/bf16 follow PRD-012b."
        )

    hash_hex = _sha256_hex(wgsl)
    spec = KernelSpec(
        wgsl=wgsl,
        name=name,
        workgroup_size=tuple(workgroup_size),
        num_inputs=num_inputs,
        output_dtype=output_dtype,
        hash=hash_hex,
    )
    # Idempotent registration. Same WGSL → same spec; re-registering is fine.
    if hash_hex in _REGISTRY:
        existing = _REGISTRY[hash_hex]
        if existing.workgroup_size != spec.workgroup_size:
            raise ValueError(
                f"custom_kernel: WGSL hash collision with different "
                f"workgroup_size — existing {existing.workgroup_size}, "
                f"new {spec.workgroup_size}. Either change the WGSL or "
                f"keep workgroup_size consistent."
            )
    else:
        _REGISTRY[hash_hex] = spec

    def builder(*inputs: Any) -> Any:
        from ._tensor_proxy import TensorProxy
        if len(inputs) != num_inputs:
            raise TypeError(
                f"custom_kernel {name!r}: expected {num_inputs} inputs, "
                f"got {len(inputs)}"
            )
        for i, inp in enumerate(inputs):
            if not isinstance(inp, TensorProxy):
                raise TypeError(
                    f"custom_kernel {name!r}: input {i} is not a TensorProxy "
                    f"(got {type(inp).__name__})"
                )
        input_shapes = [tuple(inp.shape) for inp in inputs]
        out_shape = tuple(output_shape_fn(*input_shapes))
        disp_shape = tuple(dispatch_shape_fn(*input_shapes))
        if len(disp_shape) != 3:
            raise ValueError(
                f"custom_kernel {name!r}: dispatch_shape_fn must return a "
                f"3-tuple (x, y, z); got {disp_shape}"
            )
        arg = {
            "op": "user",
            "kernel_hash": hash_hex,
            "kernel_name": name,
            "workgroup_size": spec.workgroup_size,
            "dispatch_shape": disp_shape,
            "num_inputs": num_inputs,
            "output_dtype": output_dtype,
            "output_shape": out_shape,
        }
        uop = UOp(
            op=OP_CUSTOM,
            inputs=tuple(inp._uop for inp in inputs),
            shape=out_shape,
            dtype=output_dtype,
            arg=arg,
        )
        return TensorProxy(
            uop,
            session=inputs[0]._get_session(),
            requires_grad=False,
        )

    builder.wgsl = wgsl  # expose for debugging: print(kernel.wgsl)
    builder.name = name
    builder.hash = hash_hex
    builder.spec = spec
    return builder


__all__ = ["custom_kernel", "get_registry", "KernelSpec"]
