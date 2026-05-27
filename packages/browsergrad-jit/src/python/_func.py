"""browsergrad_jit._func — functional transforms (PRD-014).

INTERNAL. Public surface lives at `bg.func.{grad, vjp, functional_call}`
and the torch shim `torch.func.*`.

Design contract per the DL/GPU review:

  * Functional `grad` MUST NOT write into `.grad`. Closure-write side
    effects don't compose with future `vmap` (which re-binds inputs to
    fresh LOAD UOps without proxy chain). The functional path realizes
    gradient UOps and returns them as `TensorProxy`s — no mutation.

  * Symbolic VJP path only. We refuse with a clear error when any UOp
    on the autograd chain lacks a registered VJP rule (vs. silently
    falling through to the closure path, which doesn't have a
    functional shape). This is the v0 honest scope — extending VJP
    coverage is PRD-014b's job.

  * No higher-order grad in v0. VJP rules emit `BROADCAST_TO` which has
    no registered VJP — second-order grad-of-grad falls off the symbolic
    path. Document the limitation; lift in a follow-up.

  * `vmap` and `jacrev` are deferred. The batching-transform requires
    per-opcode batching rules (~18 rules, ~50 LOC each = real work).
    Document the gap and tell users to loop in Python for v0.

Shipped surface:
  * `grad(fn, argnums=0)` — functional gradient. Returns gradient
    TensorProxy(s) without mutating inputs' `.grad`.
  * `vjp(fn, *inputs)` — returns (outputs, vjp_fn). `vjp_fn(seed)`
    gives the cotangent gradients.
  * `functional_call(module, params_dict, *args, **kwargs)` — stateless
    module evaluation. Required for `vmap(grad(model))` once vmap lands.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._ir import UOp, toposort, buffer as _buffer_uop, load as _load_uop
from ._errors import JitNotImplementedError, NoBackwardError
from . import _vjp


# --------------------------------------------------------------------------
# Core: functional backward — the symbolic walk without .grad mutation.
# --------------------------------------------------------------------------


def _collect_chain(root_proxy: Any) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Walk the autograd chain rooted at `root_proxy`. Returns:
       (proxy_by_uop_id, proxy_by_id) — mirrors _tensor_proxy._collect_proxies
       but local to this module so we don't import the internal helper."""
    proxy_by_uop_id: Dict[int, Any] = {}
    proxy_by_id: Dict[int, Any] = {}
    stack = [root_proxy]
    visited: set = set()
    while stack:
        p = stack.pop()
        if id(p) in visited:
            continue
        visited.add(id(p))
        proxy_by_uop_id.setdefault(id(p._uop), p)
        proxy_by_id[id(p)] = p
        if p._ctx is not None:
            for child in p._ctx.input_proxies:
                if id(child) not in visited:
                    stack.append(child)
    return proxy_by_uop_id, proxy_by_id


def _symbolic_vjp_walk(
    root_proxy: Any,
    seed_uop: UOp,
    leaf_proxies: Sequence[Any],
) -> Dict[int, UOp]:
    """Reverse-toposort walk producing gradient UOps for the listed leaves.

    `leaf_proxies` controls which inputs receive a gradient. A leaf must
    be reachable via the autograd chain from `root_proxy`; otherwise its
    slot in the returned dict is missing.

    The walk mirrors `_tensor_proxy.backward()`'s symbolic-path
    implementation but writes into a fresh dict instead of mutating
    `.grad`. Higher-order grad is dead-on-arrival until VJP rules cover
    `BROADCAST_TO` and `RESHAPE` — same caveat as backward().
    """
    proxy_by_uop_id, proxy_by_id = _collect_chain(root_proxy)
    leaf_ids = {id(p) for p in leaf_proxies}

    # Refuse if any non-leaf proxy lacks a VJP rule. Functional grad
    # cannot fall back to the closure path (which mutates .grad).
    for proxy in proxy_by_uop_id.values():
        if proxy._ctx is None:
            continue
        if _vjp.get_rule(proxy._uop.op) is None:
            raise JitNotImplementedError(
                f"bg.func.grad/vjp: opcode {proxy._uop.op!r} lacks a registered "
                f"VJP rule. Functional grad refuses to fall through to the "
                f"closure-backward path (which mutates .grad). Use "
                f"`tensor.backward()` instead, or register a VJP rule via "
                f"`bg.jit._vjp.register_vjp`."
            )

    order = toposort(root_proxy._uop)
    grad_uops: Dict[int, UOp] = {id(root_proxy): seed_uop}

    for node in reversed(order):
        proxy = proxy_by_uop_id.get(id(node))
        if proxy is None or proxy._ctx is None:
            continue
        if id(proxy) not in grad_uops:
            continue
        dy_uop = grad_uops[id(proxy)]
        rule = _vjp.get_rule(proxy._uop.op)
        input_uops = tuple(inp._uop for inp in proxy._ctx.input_proxies)
        emitted = rule(proxy._uop, input_uops, dy_uop)
        for inp_proxy, gnode in zip(proxy._ctx.input_proxies, emitted):
            if gnode is None:
                continue
            # In functional mode, propagate to every input on the chain
            # (we may stop at non-leaf intermediates if no leaf is downstream,
            # but the cost is one extra dict entry per intermediate — cheap).
            pid = id(inp_proxy)
            if pid in grad_uops:
                existing = grad_uops[pid]
                grad_uops[pid] = UOp(
                    op="ADD",
                    inputs=(existing, gnode),
                    shape=existing.shape,
                    dtype=existing.dtype,
                    arg={"vjp_of": proxy._uop, "accumulator": True},
                )
            else:
                grad_uops[pid] = gnode

    # Filter to only the requested leaves; subset is intentional.
    return {pid: grad_uops[pid] for pid in leaf_ids if pid in grad_uops}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _scalarize(out: Any) -> Any:
    """If `out` is non-scalar, reduce it via .sum() so .backward()/vjp can
    seed a 0-dim 1.0. Caller may need a different reduction — pass an
    explicit scalar then. Mirrors torch.func.grad's contract that the
    function must return a scalar."""
    if out.ndim == 0 or out.numel() == 1:
        return out
    raise RuntimeError(
        f"bg.func.grad: function must return a scalar (got shape {out.shape}). "
        f"Wrap your output with `.sum()` or `.mean()` first, or use "
        f"`bg.func.vjp(fn, ...)` for vector-valued outputs."
    )


def grad(
    fn: Callable[..., Any],
    argnums: Union[int, Tuple[int, ...]] = 0,
) -> Callable[..., Any]:
    """Functional gradient transform. Mirrors `torch.func.grad`.

    Given `fn(*args, **kwargs) -> scalar TensorProxy`, returns a callable
    that produces the gradient w.r.t. arg(s) at `argnums`. Does NOT
    mutate any input's `.grad`.

    For multiple argnums, returns a tuple of gradients in argnums order.

    Limitations (v0):
      * Function must return a scalar TensorProxy. For non-scalar outputs
        use `bg.func.vjp(fn, *args)` and apply the seed explicitly.
      * Higher-order grad (grad-of-grad) requires VJP rules for
        BROADCAST_TO/RESHAPE which aren't registered yet; raises
        JitNotImplementedError if you try.
      * Inputs at argnums must be TensorProxy. Other arg types pass
        through unchanged.
    """
    if isinstance(argnums, int):
        argnums_t: Tuple[int, ...] = (argnums,)
        single = True
    else:
        argnums_t = tuple(argnums)
        single = False

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from ._tensor_proxy import TensorProxy, from_numpy

        # The function may close over module parameters; we still expose
        # all leaf TensorProxy inputs as candidates. The argnums filter
        # selects which ones to return gradients for.
        wanted_leaves: List[Any] = []
        for i in argnums_t:
            arg_i = args[i]
            if not isinstance(arg_i, TensorProxy):
                raise TypeError(
                    f"bg.func.grad: arg at position {i} is not a TensorProxy "
                    f"(got {type(arg_i).__name__}). Wrap with bg.from_numpy or "
                    f"bg.tensor first."
                )
            wanted_leaves.append(arg_i)

        out = fn(*args, **kwargs)
        out_scalar = _scalarize(out)

        # Seed the cotangent as ones-like of the scalar. Build a BUFFER+LOAD
        # so the gradient graph is realizable.
        sess = out_scalar._get_session()
        seed_arr = np.full(
            out_scalar._uop.shape,
            1.0,
            dtype=np.dtype(out_scalar._uop.dtype),
        )
        seed_bid = sess.buffer_table.new_buffer(seed_arr)
        seed_buf = _buffer_uop(seed_bid, out_scalar._uop.shape, out_scalar._uop.dtype)
        seed_uop = _load_uop(seed_buf)

        grad_uops = _symbolic_vjp_walk(out_scalar, seed_uop, wanted_leaves)

        from ._realize import realize
        from . import _checkpoint
        out_grads: List[Any] = []
        for leaf in wanted_leaves:
            pid = id(leaf)
            if pid not in grad_uops:
                # Leaf wasn't on the autograd chain — zero gradient of
                # matching shape.
                z = np.zeros(leaf._uop.shape, dtype=np.dtype(leaf._uop.dtype))
                out_grads.append(from_numpy(z, session=sess))
                continue
            gnode = grad_uops[pid]
            if _checkpoint.has_any_region():
                gnode = _checkpoint.apply_checkpoint_rewrite(gnode)
            arr = realize(gnode, sess.buffer_table)
            out_grads.append(from_numpy(np.array(arr, copy=True), session=sess))

        return out_grads[0] if single else tuple(out_grads)

    return wrapped


def vjp(
    fn: Callable[..., Any],
    *primals: Any,
) -> Tuple[Any, Callable[[Any], Tuple[Any, ...]]]:
    """Compute outputs and a vjp function. Mirrors `torch.func.vjp`.

    `outputs, vjp_fn = vjp(fn, *primals)`
    `grads = vjp_fn(cotangent)`  → tuple of gradients, one per primal.

    Useful for vector-valued outputs where `grad` would refuse, and for
    custom Jacobian-vector products. The returned `vjp_fn` may be called
    multiple times with different cotangents; each call produces a fresh
    realize pass.

    Limitations match `grad`: VJP rule coverage required for every op on
    the chain.
    """
    from ._tensor_proxy import TensorProxy, from_numpy

    primal_proxies: List[Any] = []
    for i, p in enumerate(primals):
        if not isinstance(p, TensorProxy):
            raise TypeError(
                f"bg.func.vjp: primal at position {i} is not a TensorProxy "
                f"(got {type(p).__name__})."
            )
        primal_proxies.append(p)

    outputs = fn(*primals)
    # outputs may be a single TensorProxy or a tuple. Normalize.
    single_output = isinstance(outputs, TensorProxy)
    output_tuple: Tuple[Any, ...] = (outputs,) if single_output else tuple(outputs)

    def vjp_fn(cotangent: Any) -> Tuple[Any, ...]:
        # cotangent shape must match outputs shape. For single output,
        # accept a single TensorProxy; for tuple, accept a tuple.
        if single_output:
            cot_tuple: Tuple[Any, ...] = (cotangent,)
        else:
            cot_tuple = tuple(cotangent)
        if len(cot_tuple) != len(output_tuple):
            raise ValueError(
                f"bg.func.vjp: cotangent length {len(cot_tuple)} doesn't "
                f"match outputs length {len(output_tuple)}"
            )

        # For multi-output: sum the contributions. Reduce via a chained
        # backward where each output gets its own seed. Simplest: scalar-ize
        # via sum-of-cotangent-dot-output and call symbolic_vjp_walk once.
        # Build a synthetic "loss" = sum(cot_i * out_i).
        if len(output_tuple) == 1:
            seed_proxy = cot_tuple[0]
            # Need a scalar root for symbolic walk. Take sum(out * cotangent).
            loss = (output_tuple[0] * seed_proxy).sum()
        else:
            terms = [(o * c).sum() for o, c in zip(output_tuple, cot_tuple)]
            loss = terms[0]
            for t in terms[1:]:
                loss = loss + t

        # Now build a scalar seed of 1 and walk.
        sess = loss._get_session()
        seed_arr = np.full(loss._uop.shape, 1.0,
                           dtype=np.dtype(loss._uop.dtype))
        seed_bid = sess.buffer_table.new_buffer(seed_arr)
        seed_buf = _buffer_uop(seed_bid, loss._uop.shape, loss._uop.dtype)
        seed_uop = _load_uop(seed_buf)

        grad_uops = _symbolic_vjp_walk(loss, seed_uop, primal_proxies)
        from ._realize import realize
        from . import _checkpoint
        out_grads: List[Any] = []
        for leaf in primal_proxies:
            pid = id(leaf)
            if pid not in grad_uops:
                z = np.zeros(leaf._uop.shape,
                             dtype=np.dtype(leaf._uop.dtype))
                out_grads.append(from_numpy(z, session=sess))
                continue
            gnode = grad_uops[pid]
            if _checkpoint.has_any_region():
                gnode = _checkpoint.apply_checkpoint_rewrite(gnode)
            arr = realize(gnode, sess.buffer_table)
            out_grads.append(from_numpy(np.array(arr, copy=True), session=sess))
        return tuple(out_grads)

    return (outputs, vjp_fn)


def functional_call(
    module: Any,
    params_dict: Dict[str, Any],
    args: Union[tuple, Any] = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Stateless module evaluation. Mirrors `torch.func.functional_call`.

    Temporarily replaces `module.named_parameters()` with the values in
    `params_dict`, calls `module(*args, **kwargs)`, then restores. Used
    for MAML-style meta-learning and for `vmap(grad(model))` when vmap
    lands.

    `params_dict` keys follow PyTorch convention: `"layer1.weight"`,
    `"layer1.bias"`, etc. Nested module access via `.` separator.
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)

    # Walk module by dotted-name and collect (parent, attr_name) for each
    # parameter we'll override. Restore on exit.
    overrides: List[Tuple[Any, str, Any]] = []
    try:
        for name, new_value in params_dict.items():
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            old_value = getattr(parent, attr)
            overrides.append((parent, attr, old_value))
            setattr(parent, attr, new_value)
        return module(*args, **kwargs)
    finally:
        for parent, attr, old_value in overrides:
            setattr(parent, attr, old_value)


# --------------------------------------------------------------------------
# vmap / jacrev refuse-with-pointer
# --------------------------------------------------------------------------


def vmap(fn: Any, in_dims: Any = 0, out_dims: Any = 0) -> Any:
    """vmap — DEFERRED to PRD-014b.

    A real vmap implementation requires per-opcode batching rules (~18
    rules in our IR). For v0, this entry point exists so the surface is
    stable but refuses with a clear pointer. Use a Python `for` loop
    over the batch dimension for now — slow but correct and composable
    with `grad`.
    """
    raise JitNotImplementedError(
        "bg.func.vmap is not implemented in v0. Use a Python `for` loop "
        "over the batch dimension, or stack inputs and use the natural "
        "broadcasting behavior of the IR. A real batching transform "
        "(per-opcode rules) lands in PRD-014b."
    )


def jacrev(fn: Any, argnums: Any = 0) -> Any:
    """jacrev — DEFERRED to PRD-014b.

    `jacrev` is `vmap(grad(scalarize(f)))(I)` and depends on vmap. See
    `vmap` for the refusal pointer.
    """
    raise JitNotImplementedError(
        "bg.func.jacrev is not implemented in v0 — it depends on "
        "bg.func.vmap. Use Python-loop Jacobian assembly via bg.func.vjp "
        "with one-hot cotangents for now. Real jacrev lands in PRD-014b."
    )


__all__ = [
    "grad",
    "vjp",
    "functional_call",
    "vmap",
    "jacrev",
]
