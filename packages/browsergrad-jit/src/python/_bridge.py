"""browsergrad_jit._bridge — WebGpuBridge protocol the realizer talks to.

INTERNAL. The bridge is the sync surface between the Python realizer
(which builds IR) and the JS-side WebGPU dispatch (which owns
GPUBuffers and shader pipelines). Two implementations satisfy this
protocol:

  1. Production: `createWebGpuRealizerBridge(device)` in
     `@unlocalhosted/browsergrad-kernels`. The bridge methods are async
     on the JS side; Python calls them via JSPI (Pyodide 0.27+ on
     Chrome 144+) so Python stays sync without an async realize()
     refactor.

  2. Testing: a NumPy-backed mock implementing the same surface. Lets
     us validate the seam contract in pyodide-in-node where no
     WebGPU device exists. The mock executes each op via NumPy and
     returns an opaque handle (a per-bridge int) — Python never
     inspects what's inside.

Why a Protocol and not a class: the JS-side bridge crosses Pyodide's
boundary as a JsProxy, which doesn't subclass anything Pythonic.
Protocol typing matches structurally; runtime calls are just attribute
lookups + method invocations.

Handle opacity: the realizer treats handles as black boxes. Only the
bridge knows how to release them. The realizer's only job is to thread
them between op calls and call `materialize()` on the final root.

Op surface (v0 spike):
  - `upload(data, shape, dtype) -> handle`           — bytes → GPUBuffer
  - `materialize(handle, shape, dtype) -> bytes`     — GPUBuffer → bytes
  - `release(handle)`                                — free a GPUBuffer
  - `matmul(a, b, m, k, n, dtype) -> handle`         — A[M,K] @ B[K,N]
  - `fused_elementwise(inputs, ops, shape, dtype)`   — pre-fused chain
  - `cast(handle, src_dtype, dst_dtype, shape)`      — dtype conversion
  - `flash_attention(q, k, v, mask, b, h, sq, sk, d, scale, dtype)` — FA-v2 fwd
  - `conv1d(input, weight, bias, ..., dtype)`        — Conv1d forward
  - `conv1d_backward_input(dy, weight, ..., dtype)`  — dInput
  - `conv1d_backward_weight(dy, input, ..., dtype)`  — dWeight
  - `conv1d_backward_bias(dy, ..., dtype)`           — dBias
  - `conv2d(input, weight, bias, ..., dtype)`        — Conv2d forward
  - `conv2d_backward_input(dy, weight, ..., dtype)`  — dInput
  - `conv2d_backward_weight(dy, input, ..., dtype)`  — dWeight
  - `conv2d_backward_bias(dy, ..., dtype)`           — dBias
  - `run_tensor_plan(plan, inputs, dtype) -> bytes`  — canonical tensor IR plan
  - `run_tensor_plan_resident(plan, inputs, dtype) -> handle` — root stays on GPU

Anything outside this set raises `JitNotImplementedError` from the
realizer. The honest scope per the DL/GPU review.
"""

from __future__ import annotations
from typing import Any, Optional, Protocol, Tuple


class WebGpuBridge(Protocol):
    """Structural protocol every WebGPU bridge implements."""

    # Buffer lifecycle --------------------------------------------------
    def upload(self, data: bytes, shape: Tuple[int, ...], dtype: str) -> Any: ...

    def materialize(self, handle: Any, shape: Tuple[int, ...], dtype: str) -> bytes: ...

    def release(self, handle: Any) -> None: ...

    # Compute ops -------------------------------------------------------
    def matmul(
        self, a: Any, b: Any, m: int, k: int, n: int, dtype: str
    ) -> Any: ...

    def fused_elementwise(
        self,
        inputs: list,
        ops: list,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> Any: ...

    def cast(
        self,
        handle: Any,
        src_dtype: str,
        dst_dtype: str,
        shape: Tuple[int, ...],
    ) -> Any: ...

    def flash_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        mask: Optional[Any],
        b: int,
        h: int,
        sq: int,
        sk: int,
        d: int,
        scale: float,
        dtype: str,
    ) -> Any: ...

    def conv1d(
        self,
        input: Any,
        weight: Any,
        bias: Optional[Any],
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        k: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        l_out: int,
        dtype: str,
    ) -> Any: ...

    def conv1d_backward_input(
        self,
        dy: Any,
        weight: Any,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        k: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        l_out: int,
        dtype: str,
    ) -> Any: ...

    def conv1d_backward_weight(
        self,
        dy: Any,
        input: Any,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        k: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        l_out: int,
        dtype: str,
    ) -> Any: ...

    def conv1d_backward_bias(
        self,
        dy: Any,
        n: int,
        c_out: int,
        l_out: int,
        dtype: str,
    ) -> Any: ...

    def conv2d(
        self,
        input: Any,
        weight: Any,
        bias: Optional[Any],
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kh: int,
        kw: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        out_h: int,
        out_w: int,
        dtype: str,
    ) -> Any: ...

    def conv2d_backward_input(
        self,
        dy: Any,
        weight: Any,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kh: int,
        kw: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        out_h: int,
        out_w: int,
        dtype: str,
    ) -> Any: ...

    def conv2d_backward_weight(
        self,
        dy: Any,
        input: Any,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kh: int,
        kw: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        out_h: int,
        out_w: int,
        dtype: str,
    ) -> Any: ...

    def conv2d_backward_bias(
        self,
        dy: Any,
        n: int,
        c_out: int,
        out_h: int,
        out_w: int,
        dtype: str,
    ) -> Any: ...

    def run_user_kernel(
        self,
        inputs: list,
        wgsl: str,
        name: str,
        hash: str,
        workgroup_size: Tuple[int, int, int],
        dispatch_shape: Tuple[int, int, int],
        output_length: int,
        output_shape: Tuple[int, ...],
        dtype: str,
    ) -> Any: ...

    def run_tensor_plan(
        self,
        plan: dict,
        inputs: list,
        dtype: str,
    ) -> bytes: ...

    def run_tensor_plan_resident(
        self,
        plan: dict,
        inputs: list,
        dtype: str,
    ) -> Any: ...


__all__ = ["WebGpuBridge"]
