"""browsergrad_jit._optim — SGD and Adam optimizers.

INTERNAL. Users import as `browsergrad_jit.optim`.

Both optimizers follow PyTorch's `torch.optim` semantics:
  - Take an iterable of Parameters at construction.
  - `zero_grad()` resets every parameter's .grad to None.
  - `step()` reads each parameter's .grad and updates the parameter's
    underlying buffer in place via the BufferTable.

CPU `step()` keeps the original NumPy update path. `step(device="webgpu")`
routes supported optimizer math through primitive update IR and the tensor-plan
WebGPU bridge, then writes the materialized result back to the CPU BufferTable.
`SGD.step(device="webgpu", resident=True)` keeps the updated parameter buffer on
GPU for the no-momentum case. `Adam.step(..., resident=True)` and
`AdamW.step(..., resident=True)` keep parameter/m/v state tensors resident.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ._ir import (
    UOp,
    OP_SGD_UPDATE,
    OP_ADAMW_UPDATE_M,
    OP_ADAMW_UPDATE_V,
    OP_ADAMW_UPDATE_PARAM,
    OP_ADAM_UPDATE_M,
    OP_ADAM_UPDATE_V,
    OP_ADAM_UPDATE_PARAM,
)
from ._tensor_proxy import (
    TensorProxy,
    from_numpy,
    _from_buffer_id,
    _has_gpu_resident_inputs,
)
from ._errors import RealizationError, ShapeError


def _normalize_step_device(device: Optional[str]) -> str:
    if device is None:
        return "cpu"
    out = str(device).lower()
    if out in ("gpu", "tensor_plan_webgpu"):
        out = "webgpu"
    if out not in ("cpu", "webgpu"):
        raise ValueError(
            f"optimizer.step(device=...): expected 'cpu' or 'webgpu', got {device!r}"
        )
    return out


def _tensor_has_gpu_residency(tensor: Optional[TensorProxy]) -> bool:
    if tensor is None:
        return False
    try:
        sess = tensor._get_session()
        return _has_gpu_resident_inputs(tensor._uop, sess)
    except Exception:
        return False


def _resolve_step_device_and_residency(
    params: List[TensorProxy],
    *,
    device: Optional[str],
    resident: bool,
    name: str,
) -> tuple[str, bool]:
    if device is None:
        if resident:
            return "webgpu", True
        for p in params:
            if _tensor_has_gpu_residency(p) or _tensor_has_gpu_residency(p.grad):
                return "webgpu", True
        return "cpu", False
    step_device = _normalize_step_device(device)
    if resident and step_device != "webgpu":
        raise ValueError(f"{name}.step(resident=True) requires device='webgpu'")
    return step_device, resident


def _realize_update_webgpu(tensor: TensorProxy) -> np.ndarray:
    from ._realize_webgpu import get_registered_gpu_buffer_table, realize_tensor_plan_webgpu
    gbt = get_registered_gpu_buffer_table()
    if gbt is None:
        raise RealizationError(
            "optimizer.step(device='webgpu') requires a registered WebGPU bridge. "
            "Call browsergrad_jit.register_webgpu_bridge(...) first."
        )
    return realize_tensor_plan_webgpu(
        tensor._uop,
        numpy_buffer_table=tensor._get_session().buffer_table,
        gpu_buffer_table=gbt,
    )


def _realize_update_webgpu_resident(tensor: TensorProxy) -> TensorProxy:
    from ._realize_webgpu import (
        get_registered_gpu_buffer_table,
        realize_tensor_plan_webgpu_resident,
    )
    gbt = get_registered_gpu_buffer_table()
    if gbt is None:
        raise RealizationError(
            "optimizer.step(device='webgpu', resident=True) requires a registered "
            "WebGPU bridge. Call browsergrad_jit.register_webgpu_bridge(...) first."
        )
    sess = tensor._get_session()
    tmp_bid = realize_tensor_plan_webgpu_resident(
        tensor._uop,
        numpy_buffer_table=sess.buffer_table,
        gpu_buffer_table=gbt,
    )
    return _from_buffer_id(tmp_bid, tensor.shape, tensor.dtype, session=sess)


def _replace_param_with_webgpu_resident_tensor(p: TensorProxy, updated: TensorProxy) -> None:
    from ._realize_webgpu import get_registered_gpu_buffer_table
    gbt = get_registered_gpu_buffer_table()
    if gbt is None:
        raise RealizationError(
            "optimizer.step(device='webgpu', resident=True) requires a registered "
            "WebGPU bridge. Call browsergrad_jit.register_webgpu_bridge(...) first."
        )
    sess = p._get_session()
    param_bid = _param_buffer_id(p)
    tmp_bid = _param_buffer_id(updated)
    handle = gbt.detach(tmp_bid)
    sess.buffer_table.evict(tmp_bid)
    sess.buffer_table.mark_unmaterialized(param_bid)
    gbt.replace(param_bid, handle)


def _replace_param_with_webgpu_resident_update(p: TensorProxy, updated: TensorProxy) -> None:
    _replace_param_with_webgpu_resident_tensor(p, _realize_update_webgpu_resident(updated))


def _param_buffer_id(p: TensorProxy) -> str:
    """Extract the underlying BUFFER's id from a Parameter (which is a
    TensorProxy wrapping LOAD(BUFFER))."""
    uop = p._uop
    if uop.op == "LOAD" and len(uop.inputs) == 1 and uop.inputs[0].op == "BUFFER":
        return uop.inputs[0].arg
    raise RealizationError(
        f"optimizer: parameter is not a LOAD-of-BUFFER (op={uop.op}); "
        f"optimizers operate only on leaf parameters."
    )


def sgd_update(
    param: TensorProxy,
    grad: TensorProxy,
    *,
    lr: float,
    weight_decay: float = 0.0,
) -> TensorProxy:
    """Functional SGD update node: `param - lr * (grad + wd * param)`.

    This is the optimizer/update IR primitive for GPU tensor plans. It does
    not mutate `param`; optimizers can later pair it with STORE/resident
    buffer state when the runtime owns in-place updates end-to-end.
    """
    if param.shape != grad.shape:
        raise ShapeError(
            f"sgd_update: param shape {param.shape} must match grad shape {grad.shape}"
        )
    if param.dtype != grad.dtype:
        raise ShapeError(
            f"sgd_update: param dtype {param.dtype} must match grad dtype {grad.dtype}"
        )
    if lr < 0:
        raise ValueError(f"sgd_update: lr must be >= 0, got {lr}")
    uop = UOp(
        op=OP_SGD_UPDATE,
        inputs=(param._uop, grad._uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={"lr": float(lr), "weight_decay": float(weight_decay)},
    )
    return TensorProxy(uop, session=param._get_session(), requires_grad=False)


def adamw_update(
    param: TensorProxy,
    grad: TensorProxy,
    m: TensorProxy,
    v: TensorProxy,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    """Functional AdamW update IR.

    Returns `(new_param, new_m, new_v)`. No mutation. This lets tensor-plan
    WebGPU keep params/grad/state as graph values before the runtime grows
    resident in-place optimizer state.
    """
    for name, tensor in (("grad", grad), ("m", m), ("v", v)):
        if tensor.shape != param.shape:
            raise ShapeError(
                f"adamw_update: {name} shape {tensor.shape} must match param shape {param.shape}"
            )
        if tensor.dtype != param.dtype:
            raise ShapeError(
                f"adamw_update: {name} dtype {tensor.dtype} must match param dtype {param.dtype}"
            )
    if lr < 0:
        raise ValueError(f"adamw_update: lr must be >= 0, got {lr}")
    if step <= 0:
        raise ValueError(f"adamw_update: step must be >= 1, got {step}")
    beta1, beta2 = float(betas[0]), float(betas[1])
    sess = param._get_session()
    m_uop = UOp(
        op=OP_ADAMW_UPDATE_M,
        inputs=(m._uop, grad._uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={"beta1": beta1},
    )
    v_uop = UOp(
        op=OP_ADAMW_UPDATE_V,
        inputs=(v._uop, grad._uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={"beta2": beta2},
    )
    p_uop = UOp(
        op=OP_ADAMW_UPDATE_PARAM,
        inputs=(param._uop, grad._uop, m_uop, v_uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={
            "lr": float(lr),
            "beta1": beta1,
            "beta2": beta2,
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "step": int(step),
        },
    )
    return (
        TensorProxy(p_uop, session=sess, requires_grad=False),
        TensorProxy(m_uop, session=sess, requires_grad=False),
        TensorProxy(v_uop, session=sess, requires_grad=False),
    )


def adam_update(
    param: TensorProxy,
    grad: TensorProxy,
    m: TensorProxy,
    v: TensorProxy,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    """Functional Adam update IR with coupled weight decay.

    Returns `(new_param, new_m, new_v)`. Unlike AdamW, Adam's weight decay is
    added to the gradient before the moment updates, matching PyTorch's
    coupled-decay semantics.
    """
    for name, tensor in (("grad", grad), ("m", m), ("v", v)):
        if tensor.shape != param.shape:
            raise ShapeError(
                f"adam_update: {name} shape {tensor.shape} must match param shape {param.shape}"
            )
        if tensor.dtype != param.dtype:
            raise ShapeError(
                f"adam_update: {name} dtype {tensor.dtype} must match param dtype {param.dtype}"
            )
    if lr < 0:
        raise ValueError(f"adam_update: lr must be >= 0, got {lr}")
    if step <= 0:
        raise ValueError(f"adam_update: step must be >= 1, got {step}")
    beta1, beta2 = float(betas[0]), float(betas[1])
    sess = param._get_session()
    m_uop = UOp(
        op=OP_ADAM_UPDATE_M,
        inputs=(param._uop, grad._uop, m._uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={"beta1": beta1, "weight_decay": float(weight_decay)},
    )
    v_uop = UOp(
        op=OP_ADAM_UPDATE_V,
        inputs=(param._uop, grad._uop, v._uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={"beta2": beta2, "weight_decay": float(weight_decay)},
    )
    p_uop = UOp(
        op=OP_ADAM_UPDATE_PARAM,
        inputs=(param._uop, m_uop, v_uop),
        shape=param.shape,
        dtype=param.dtype,
        arg={
            "lr": float(lr),
            "beta1": beta1,
            "beta2": beta2,
            "eps": float(eps),
            "step": int(step),
        },
    )
    return (
        TensorProxy(p_uop, session=sess, requires_grad=False),
        TensorProxy(m_uop, session=sess, requires_grad=False),
        TensorProxy(v_uop, session=sess, requires_grad=False),
    )


class Optimizer:
    """Base — minimal protocol matching torch.optim.Optimizer."""

    def __init__(self, params: Iterable[TensorProxy]) -> None:
        self._params: List[TensorProxy] = list(params)
        if not self._params:
            raise ValueError("optimizer: parameter list is empty")
        for p in self._params:
            if not p.requires_grad:
                # Allow non-grad params to coexist (e.g. frozen layers), but
                # never step them. Matching PyTorch's behavior — frozen params
                # have requires_grad=False and step skips them.
                pass

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def step(self, device: Optional[str] = None, resident: bool = False) -> None:
        raise NotImplementedError

    def _group_options(self) -> Dict[str, Any]:
        return {}

    def _serialize_state(self) -> Dict[int, Dict[str, Any]]:
        return {}

    def _load_serialized_state(
        self,
        state: Dict[int, Dict[str, Any]],
        options: Dict[str, Any],
    ) -> None:
        if state or options:
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: unsupported state"
            )

    def state_dict(self) -> Dict[str, Any]:
        """Return portable single-group optimizer state using positional ids."""
        group = {
            **self._group_options(),
            "params": list(range(len(self._params))),
        }
        return {
            "state": self._serialize_state(),
            "param_groups": [group],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore a compatible optimizer state without partial mutation."""
        if not isinstance(state_dict, dict):
            raise TypeError(
                "optimizer.load_state_dict: expected dict, "
                f"got {type(state_dict).__name__}"
            )
        if set(state_dict) != {"state", "param_groups"}:
            raise ValueError(
                "optimizer.load_state_dict: expected exactly 'state' and "
                "'param_groups'"
            )
        raw_state = state_dict["state"]
        raw_groups = state_dict["param_groups"]
        if not isinstance(raw_state, dict):
            raise TypeError("optimizer.load_state_dict: state must be a dict")
        if any(type(identifier) is not int for identifier in raw_state):
            raise TypeError(
                "optimizer.load_state_dict: state keys must be integer "
                "parameter ids"
            )
        if not isinstance(raw_groups, (list, tuple)) or len(raw_groups) != 1:
            raise ValueError(
                "optimizer.load_state_dict: BrowserGrad supports exactly one "
                "parameter group"
            )
        raw_group = raw_groups[0]
        if not isinstance(raw_group, dict) or "params" not in raw_group:
            raise TypeError(
                "optimizer.load_state_dict: parameter group must be a dict "
                "containing params"
            )
        serialized_ids = raw_group["params"]
        if not isinstance(serialized_ids, (list, tuple)):
            raise TypeError(
                "optimizer.load_state_dict: param group params must be a sequence"
            )
        if len(serialized_ids) != len(self._params):
            raise ValueError(
                "optimizer.load_state_dict: parameter count mismatch: "
                f"checkpoint has {len(serialized_ids)}, optimizer has "
                f"{len(self._params)}"
            )
        if any(type(identifier) is not int for identifier in serialized_ids):
            raise TypeError(
                "optimizer.load_state_dict: serialized parameter ids must be integers"
            )
        if len(set(serialized_ids)) != len(serialized_ids):
            raise ValueError(
                "optimizer.load_state_dict: serialized parameter ids must be unique"
            )
        positions = {identifier: index for index, identifier in enumerate(serialized_ids)}
        unknown = set(raw_state) - set(positions)
        if unknown:
            raise ValueError(
                "optimizer.load_state_dict: state references unknown parameter "
                f"ids {sorted(unknown)!r}"
            )
        mapped_state = {
            positions[identifier]: value
            for identifier, value in raw_state.items()
        }
        if any(not isinstance(value, dict) for value in mapped_state.values()):
            raise TypeError(
                "optimizer.load_state_dict: every parameter state must be a dict"
            )
        options = {key: value for key, value in raw_group.items() if key != "params"}
        self._load_serialized_state(mapped_state, options)


class SGD(Optimizer):
    """Standard SGD with optional momentum.

    Mirrors `torch.optim.SGD(params, lr, momentum=0)`.
    """

    def __init__(
        self,
        params: Iterable[TensorProxy],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"SGD: lr must be >= 0, got {lr}")
        if momentum < 0:
            raise ValueError(f"SGD: momentum must be >= 0, got {momentum}")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Momentum buffers per parameter — indexed by Parameter identity.
        self._velocity: dict[int, np.ndarray] = {}

    def _group_options(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }

    def _serialize_state(self) -> Dict[int, Dict[str, Any]]:
        state = {}
        for index, parameter in enumerate(self._params):
            velocity = self._velocity.get(id(parameter))
            if velocity is not None:
                state[index] = {
                    "momentum_buffer": np.array(velocity, copy=True),
                }
        return state

    def _load_serialized_state(
        self,
        state: Dict[int, Dict[str, Any]],
        options: Dict[str, Any],
    ) -> None:
        expected_options = {"lr", "momentum", "weight_decay"}
        if set(options) != expected_options:
            raise ValueError(
                "SGD.load_state_dict: expected group options "
                f"{sorted(expected_options)!r}, got {sorted(options)!r}"
            )
        try:
            lr = float(options["lr"])
            momentum = float(options["momentum"])
            weight_decay = float(options["weight_decay"])
        except (TypeError, ValueError) as error:
            raise TypeError("SGD.load_state_dict: options must be numeric") from error
        if lr < 0 or momentum < 0:
            raise ValueError("SGD.load_state_dict: lr and momentum must be non-negative")

        velocity = {}
        for index, item in state.items():
            if set(item) != {"momentum_buffer"}:
                raise ValueError(
                    "SGD.load_state_dict: parameter state must contain only "
                    "momentum_buffer"
                )
            array = item["momentum_buffer"]
            parameter = self._params[index]
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    "SGD.load_state_dict: momentum_buffer must be an ndarray"
                )
            if array.shape != parameter.shape or array.dtype.name != parameter.dtype:
                raise ShapeError(
                    "SGD.load_state_dict: momentum_buffer metadata does not "
                    f"match parameter {index}"
                )
            velocity[id(parameter)] = np.array(array, copy=True)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = velocity

    def step(self, device: Optional[str] = None, resident: bool = False) -> None:
        step_device, resident = _resolve_step_device_and_residency(
            self._params,
            device=device,
            resident=resident,
            name="SGD",
        )
        if step_device == "webgpu":
            if self.momentum != 0.0:
                raise RealizationError(
                    "SGD.step(device='webgpu') does not support momentum yet. "
                    "Use momentum=0, CPU step(), or functional optimizer IR."
                )
            for p in self._params:
                if not p.requires_grad or p.grad is None:
                    continue
                updated = sgd_update(
                    p,
                    p.grad,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                if resident:
                    _replace_param_with_webgpu_resident_update(p, updated)
                    continue
                bid = _param_buffer_id(p)
                sess = p._get_session()
                current = sess.buffer_table.get(bid)
                new_value = _realize_update_webgpu(updated)
                sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))
            return

        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()  # realize the gradient
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.numpy()
            if self.momentum != 0.0:
                if id(p) not in self._velocity:
                    self._velocity[id(p)] = np.zeros_like(grad)
                self._velocity[id(p)] = self.momentum * self._velocity[id(p)] + grad
                update = self._velocity[id(p)]
            else:
                update = grad
            # In-place buffer update via the session's BufferTable.
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            new_value = current - self.lr * update
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


class Adam(Optimizer):
    """Adam optimizer matching torch.optim.Adam defaults."""

    def __init__(
        self,
        params: Iterable[TensorProxy],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"Adam: lr must be >= 0, got {lr}")
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._m_resident: dict[int, TensorProxy] = {}
        self._v_resident: dict[int, TensorProxy] = {}

    def _group_options(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step": self._step,
        }

    def _serialize_state(self) -> Dict[int, Dict[str, Any]]:
        state = {}
        for index, parameter in enumerate(self._params):
            parameter_id = id(parameter)
            if parameter_id in self._m_resident:
                first = self._m_resident[parameter_id].numpy()
                second = self._v_resident[parameter_id].numpy()
            elif parameter_id in self._m:
                first = self._m[parameter_id]
                second = self._v[parameter_id]
            else:
                continue
            state[index] = {
                "step": self._step,
                "exp_avg": np.array(first, copy=True),
                "exp_avg_sq": np.array(second, copy=True),
            }
        return state

    def _load_serialized_state(
        self,
        state: Dict[int, Dict[str, Any]],
        options: Dict[str, Any],
    ) -> None:
        expected_options = {"lr", "betas", "eps", "weight_decay", "step"}
        if set(options) != expected_options:
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: expected group options "
                f"{sorted(expected_options)!r}, got {sorted(options)!r}"
            )
        betas = options["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise TypeError(
                f"{type(self).__name__}.load_state_dict: betas must have length 2"
            )
        try:
            lr = float(options["lr"])
            beta1 = float(betas[0])
            beta2 = float(betas[1])
            eps = float(options["eps"])
            weight_decay = float(options["weight_decay"])
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{type(self).__name__}.load_state_dict: options must be numeric"
            ) from error
        step = options["step"]
        if type(step) is not int or step < 0:
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: step must be a non-negative int"
            )
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: lr, eps, and "
                "weight_decay must be non-negative"
            )
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: betas must be in [0, 1)"
            )

        first_moment = {}
        second_moment = {}
        for index, item in state.items():
            if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(
                    f"{type(self).__name__}.load_state_dict: parameter state "
                    "must contain step, exp_avg, and exp_avg_sq"
                )
            if type(item["step"]) is not int or item["step"] != step:
                raise ValueError(
                    f"{type(self).__name__}.load_state_dict: per-parameter "
                    "steps must match the group step"
                )
            parameter = self._params[index]
            exp_avg = item["exp_avg"]
            exp_avg_sq = item["exp_avg_sq"]
            if not isinstance(exp_avg, np.ndarray) or not isinstance(exp_avg_sq, np.ndarray):
                raise TypeError(
                    f"{type(self).__name__}.load_state_dict: moments must be ndarrays"
                )
            for name, array in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
                if array.shape != parameter.shape or array.dtype.name != parameter.dtype:
                    raise ShapeError(
                        f"{type(self).__name__}.load_state_dict: {name} metadata "
                        f"does not match parameter {index}"
                    )
            first_moment[id(parameter)] = np.array(exp_avg, copy=True)
            second_moment[id(parameter)] = np.array(exp_avg_sq, copy=True)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = step
        self._m = first_moment
        self._v = second_moment
        self._m_resident = {}
        self._v_resident = {}

    def step(self, device: Optional[str] = None, resident: bool = False) -> None:
        self._step += 1
        step_device, resident = _resolve_step_device_and_residency(
            self._params,
            device=device,
            resident=resident,
            name="Adam",
        )
        if resident:
            for p in self._params:
                if not p.requires_grad or p.grad is None:
                    continue
                bid = _param_buffer_id(p)
                sess = p._get_session()
                shape, dtype = sess.buffer_table.metadata(bid)
                pid = id(p)
                if pid not in self._m_resident:
                    self._m_resident[pid] = from_numpy(
                        np.zeros(shape, dtype=np.dtype(dtype)),
                        session=sess,
                    )
                    self._v_resident[pid] = from_numpy(
                        np.zeros(shape, dtype=np.dtype(dtype)),
                        session=sess,
                    )
                new_p, new_m, new_v = adam_update(
                    p,
                    p.grad,
                    self._m_resident[pid],
                    self._v_resident[pid],
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    step=self._step,
                )
                p_resident = _realize_update_webgpu_resident(new_p)
                self._m_resident[pid] = _realize_update_webgpu_resident(new_m)
                self._v_resident[pid] = _realize_update_webgpu_resident(new_v)
                self._m.pop(pid, None)
                self._v.pop(pid, None)
                _replace_param_with_webgpu_resident_tensor(p, p_resident)
            return
        if step_device == "webgpu":
            for p in self._params:
                if not p.requires_grad or p.grad is None:
                    continue
                bid = _param_buffer_id(p)
                sess = p._get_session()
                current = sess.buffer_table.get(bid)
                pid = id(p)
                if pid not in self._m:
                    if pid in self._m_resident:
                        self._m[pid] = self._m_resident[pid].numpy()
                        self._v[pid] = self._v_resident[pid].numpy()
                    else:
                        self._m[pid] = np.zeros_like(current)
                        self._v[pid] = np.zeros_like(current)
                self._m_resident.pop(pid, None)
                self._v_resident.pop(pid, None)
                m = from_numpy(self._m[pid], session=sess)
                v = from_numpy(self._v[pid], session=sess)
                new_p, new_m, new_v = adam_update(
                    p,
                    p.grad,
                    m,
                    v,
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    step=self._step,
                )
                p_arr = _realize_update_webgpu(new_p)
                m_arr = _realize_update_webgpu(new_m)
                v_arr = _realize_update_webgpu(new_v)
                self._m[pid] = m_arr.astype(current.dtype, copy=False)
                self._v[pid] = v_arr.astype(current.dtype, copy=False)
                sess.buffer_table.update(bid, p_arr.astype(current.dtype, copy=False))
            return

        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.numpy()
            if id(p) not in self._m:
                if id(p) in self._m_resident:
                    self._m[id(p)] = self._m_resident[id(p)].numpy()
                    self._v[id(p)] = self._v_resident[id(p)].numpy()
                else:
                    self._m[id(p)] = np.zeros_like(grad)
                    self._v[id(p)] = np.zeros_like(grad)
            self._m_resident.pop(id(p), None)
            self._v_resident.pop(id(p), None)
            self._m[id(p)] = self.beta1 * self._m[id(p)] + (1 - self.beta1) * grad
            self._v[id(p)] = self.beta2 * self._v[id(p)] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[id(p)] / (1 - self.beta1 ** self._step)
            v_hat = self._v[id(p)] / (1 - self.beta2 ** self._step)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            new_value = current - self.lr * update
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


class AdamW(Adam):
    """Adam with decoupled weight decay (the right Adam most papers actually
    use). Matches torch.optim.AdamW."""

    def step(self, device: Optional[str] = None, resident: bool = False) -> None:
        self._step += 1
        step_device, resident = _resolve_step_device_and_residency(
            self._params,
            device=device,
            resident=resident,
            name="AdamW",
        )
        if resident:
            for p in self._params:
                if not p.requires_grad or p.grad is None:
                    continue
                bid = _param_buffer_id(p)
                sess = p._get_session()
                shape, dtype = sess.buffer_table.metadata(bid)
                pid = id(p)
                if pid not in self._m_resident:
                    self._m_resident[pid] = from_numpy(
                        np.zeros(shape, dtype=np.dtype(dtype)),
                        session=sess,
                    )
                    self._v_resident[pid] = from_numpy(
                        np.zeros(shape, dtype=np.dtype(dtype)),
                        session=sess,
                    )
                new_p, new_m, new_v = adamw_update(
                    p,
                    p.grad,
                    self._m_resident[pid],
                    self._v_resident[pid],
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    step=self._step,
                )
                p_resident = _realize_update_webgpu_resident(new_p)
                self._m_resident[pid] = _realize_update_webgpu_resident(new_m)
                self._v_resident[pid] = _realize_update_webgpu_resident(new_v)
                self._m.pop(pid, None)
                self._v.pop(pid, None)
                _replace_param_with_webgpu_resident_tensor(p, p_resident)
            return
        if step_device == "webgpu":
            for p in self._params:
                if not p.requires_grad or p.grad is None:
                    continue
                bid = _param_buffer_id(p)
                sess = p._get_session()
                current = sess.buffer_table.get(bid)
                pid = id(p)
                if pid not in self._m:
                    if pid in self._m_resident:
                        self._m[pid] = self._m_resident[pid].numpy()
                        self._v[pid] = self._v_resident[pid].numpy()
                    else:
                        self._m[pid] = np.zeros_like(current)
                        self._v[pid] = np.zeros_like(current)
                self._m_resident.pop(pid, None)
                self._v_resident.pop(pid, None)
                m = from_numpy(self._m[pid], session=sess)
                v = from_numpy(self._v[pid], session=sess)
                new_p, new_m, new_v = adamw_update(
                    p,
                    p.grad,
                    m,
                    v,
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    step=self._step,
                )
                p_arr = _realize_update_webgpu(new_p)
                m_arr = _realize_update_webgpu(new_m)
                v_arr = _realize_update_webgpu(new_v)
                self._m[pid] = m_arr.astype(current.dtype, copy=False)
                self._v[pid] = v_arr.astype(current.dtype, copy=False)
                sess.buffer_table.update(bid, p_arr.astype(current.dtype, copy=False))
            return

        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()
            if id(p) not in self._m:
                if id(p) in self._m_resident:
                    self._m[id(p)] = self._m_resident[id(p)].numpy()
                    self._v[id(p)] = self._v_resident[id(p)].numpy()
                else:
                    self._m[id(p)] = np.zeros_like(grad)
                    self._v[id(p)] = np.zeros_like(grad)
            self._m_resident.pop(id(p), None)
            self._v_resident.pop(id(p), None)
            self._m[id(p)] = self.beta1 * self._m[id(p)] + (1 - self.beta1) * grad
            self._v[id(p)] = self.beta2 * self._v[id(p)] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[id(p)] / (1 - self.beta1 ** self._step)
            v_hat = self._v[id(p)] / (1 - self.beta2 ** self._step)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            # AdamW: weight decay decoupled, applied directly to parameters.
            new_value = current - self.lr * update - self.lr * self.weight_decay * current
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "sgd_update",
    "adam_update",
    "adamw_update",
]
