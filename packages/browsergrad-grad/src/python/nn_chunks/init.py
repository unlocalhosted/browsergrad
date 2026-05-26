# ─── nn.init namespace ─────────────────────────────────────
# In-place initializers matching torch.nn.init.*. They mutate the tensor's
# .data buffer directly — the tensor identity (and any registered-param
# wiring) is preserved.

import math as _bg_init_math
import types as _bg_init_types
init = _bg_init_types.ModuleType("browsergrad_grad.nn.init")

def _init_zeros_(t):
    t.data[...] = 0.0
    return t
def _init_ones_(t):
    t.data[...] = 1.0
    return t
def _init_uniform_(t, a=0.0, b=1.0):
    t.data[...] = np.random.uniform(a, b, size=t.data.shape).astype(np.float32)
    return t
def _init_normal_(t, mean=0.0, std=1.0):
    t.data[...] = np.random.normal(mean, std, size=t.data.shape).astype(np.float32)
    return t
def _init_constant_(t, val):
    t.data[...] = float(val)
    return t
def _init_kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"):
    # Compute fan_in from the tensor shape. For Linear-like (out, in)
    # or Conv (out, in, *spatial), fan_in = in * prod(spatial).
    if t.data.ndim < 2:
        fan_in = t.data.shape[0] if t.data.ndim > 0 else 1
    else:
        fan_in = int(np.prod(t.data.shape[1:]))
    gain = _bg_init_math.sqrt(2.0 / (1.0 + a * a)) if nonlinearity == "leaky_relu" else _bg_init_math.sqrt(2.0)
    std = gain / _bg_init_math.sqrt(fan_in)
    bound = _bg_init_math.sqrt(3.0) * std
    t.data[...] = np.random.uniform(-bound, bound, size=t.data.shape).astype(np.float32)
    return t
def _init_xavier_uniform_(t, gain=1.0):
    if t.data.ndim < 2:
        fan_in = fan_out = t.data.shape[0] if t.data.ndim > 0 else 1
    else:
        fan_out = t.data.shape[0]
        fan_in = int(np.prod(t.data.shape[1:]))
    bound = gain * _bg_init_math.sqrt(6.0 / (fan_in + fan_out))
    t.data[...] = np.random.uniform(-bound, bound, size=t.data.shape).astype(np.float32)
    return t

init.zeros_ = _init_zeros_
init.ones_ = _init_ones_
init.uniform_ = _init_uniform_
init.normal_ = _init_normal_
init.constant_ = _init_constant_
init.kaiming_uniform_ = _init_kaiming_uniform_
init.xavier_uniform_ = _init_xavier_uniform_

# Register under sys.modules so 'from browsergrad_grad.nn import init'
# works AND 'import browsergrad_grad.nn.init' resolves.
import sys as _bg_init_sys
_bg_init_sys.modules["browsergrad_grad.nn.init"] = init
# Leave _bg_init_math / _bg_init_types / _bg_init_sys in module globals —
# the init functions reference them as free vars at call time. The single
# underscore prefix already keeps them out of star-imports.
