import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("JIT CNN modules and functionals", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("conv1d supports groups, dilation, primitive IR, and symbolic backward", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      forwardClose: boolean;
      outShape: number[];
      op: string;
      gradShapes: number[][];
      gradNonzero: boolean[];
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
import browsergrad_jit.nn.functional as F
from browsergrad_jit._ir import toposort

def conv1d_ref(x, w, b, stride=1, padding=0, dilation=1, groups=1):
    N, C_in, L = x.shape
    C_out, cpg, K = w.shape
    x_pad = np.pad(x, ((0,0),(0,0),(padding,padding)), mode="constant")
    L_out = (L + 2*padding - (dilation*(K-1)+1)) // stride + 1
    out = np.zeros((N, C_out, L_out), dtype=np.float32)
    opg = C_out // groups
    for n in range(N):
        for g in range(groups):
            c0 = g * cpg
            o0 = g * opg
            for co in range(opg):
                for i in range(L_out):
                    l0 = i * stride
                    out[n, o0+co, i] = (
                        x_pad[n, c0:c0+cpg, l0:l0+dilation*(K-1)+1:dilation]
                        * w[o0+co]
                    ).sum()
    return out + b.reshape(1, C_out, 1)

x_data = np.arange(1, 1 + 1*4*9, dtype=np.float32).reshape(1, 4, 9) / 10
w_data = np.arange(1, 1 + 6*2*3, dtype=np.float32).reshape(6, 2, 3) / 20
b_data = np.linspace(-0.3, 0.3, 6, dtype=np.float32)
x = bg.tensor(x_data, requires_grad=True)
w = bg.tensor(w_data, requires_grad=True)
b = bg.tensor(b_data, requires_grad=True)
y = F.conv1d(x, w, b, stride=2, padding=2, dilation=2, groups=2)
expected = conv1d_ref(x_data, w_data, b_data, stride=2, padding=2, dilation=2, groups=2)
y.sum().backward()
gx = bg.func.grad(lambda inp: F.conv1d(inp, w, b, stride=2, padding=2, dilation=2, groups=2).sum())(x)
gw = bg.func.grad(lambda weight: F.conv1d(x, weight, b, stride=2, padding=2, dilation=2, groups=2).sum())(w)
gb = bg.func.grad(lambda bias: F.conv1d(x, w, bias, stride=2, padding=2, dilation=2, groups=2).sum())(b)
{
    "forwardClose": bool(np.allclose(y.numpy(), expected, atol=1e-5)),
    "outShape": list(y.shape),
    "op": y._uop.op,
    "gradShapes": [list(x.grad.shape), list(w.grad.shape), list(b.grad.shape)],
    "gradNonzero": [
        bool(np.abs(x.grad.numpy()).sum() > 0),
        bool(np.abs(w.grad.numpy()).sum() > 0),
        bool(np.abs(b.grad.numpy()).sum() > 0),
    ],
    "gxOps": [u.op for u in toposort(gx._uop)],
    "gwOps": [u.op for u in toposort(gw._uop)],
    "gbOps": [u.op for u in toposort(gb._uop)],
}
`);
    expect(result.forwardClose).toBe(true);
    expect(result.outShape).toEqual([1, 6, 5]);
    expect(result.op).toBe("CONV1D");
    expect(result.gradShapes).toEqual([[1, 4, 9], [6, 2, 3], [6]]);
    expect(result.gradNonzero).toEqual([true, true, true]);
    expect(result.gxOps).toContain("CONV1D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV1D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV1D_BACKWARD_BIAS");
  });

  it("conv2d supports tuple stride/padding, dilation, groups, and backward", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      forwardClose: boolean;
      outShape: number[];
      op: string;
      gradShapes: number[][];
      gradNonzero: boolean[];
      keys: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
import browsergrad_jit.nn.functional as F

def conv2d_ref(x, w, b, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
    N, C_in, H, W = x.shape
    C_out, cpg, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    x_pad = np.pad(x, ((0,0),(0,0),(ph,ph),(pw,pw)), mode="constant")
    H_out = (H + 2*ph - (dh*(kh-1)+1)) // sh + 1
    W_out = (W + 2*pw - (dw*(kw-1)+1)) // sw + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    opg = C_out // groups
    for n in range(N):
        for g in range(groups):
            c0 = g * cpg
            o0 = g * opg
            for co in range(opg):
                for i in range(H_out):
                    for j in range(W_out):
                        h0, w0 = i * sh, j * sw
                        out[n, o0+co, i, j] = (
                            x_pad[n, c0:c0+cpg, h0:h0+dh*(kh-1)+1:dh, w0:w0+dw*(kw-1)+1:dw]
                            * w[o0+co]
                        ).sum()
    return out + b.reshape(1, C_out, 1, 1)

x_data = np.arange(1, 1 + 1*4*5*4, dtype=np.float32).reshape(1, 4, 5, 4) / 10
w_data = np.arange(1, 1 + 6*2*2*2, dtype=np.float32).reshape(6, 2, 2, 2) / 20
b_data = np.linspace(-0.3, 0.3, 6, dtype=np.float32)
x = bg.tensor(x_data, requires_grad=True)
w = bg.tensor(w_data, requires_grad=True)
b = bg.tensor(b_data, requires_grad=True)
y = F.conv2d(x, w, b, stride=(1,1), padding=(2,0), dilation=(2,1), groups=2)
expected = conv2d_ref(x_data, w_data, b_data, stride=(1,1), padding=(2,0), dilation=(2,1), groups=2)
y.sum().backward()
m = bg.nn.Conv2d(4, 6, (2, 2), stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=2)
{
    "forwardClose": bool(np.allclose(y.numpy(), expected, atol=1e-5)),
    "outShape": list(y.shape),
    "op": y._uop.op,
    "gradShapes": [list(x.grad.shape), list(w.grad.shape), list(b.grad.shape)],
    "gradNonzero": [
        bool(np.abs(x.grad.numpy()).sum() > 0),
        bool(np.abs(w.grad.numpy()).sum() > 0),
        bool(np.abs(b.grad.numpy()).sum() > 0),
    ],
    "keys": sorted(m.state_dict().keys()),
}
`);
    expect(result.forwardClose).toBe(true);
    expect(result.outShape).toEqual([1, 6, 7, 3]);
    expect(result.op).toBe("CONV2D");
    expect(result.gradShapes).toEqual([[1, 4, 5, 4], [6, 2, 2, 2], [6]]);
    expect(result.gradNonzero).toEqual([true, true, true]);
    expect(result.keys).toEqual(["bias", "weight"]);
  });

  it("conv2d exposes symbolic backward IR for functional grad", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      gxClose: boolean;
      gwClose: boolean;
      gbClose: boolean;
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
      registered: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
import browsergrad_jit.nn.functional as F
from browsergrad_jit._ir import toposort
from browsergrad_jit._vjp import list_registered

x_data = np.arange(1, 1 + 1*4*4*3, dtype=np.float32).reshape(1, 4, 4, 3) / 13
w_data = np.arange(1, 1 + 6*2*2*2, dtype=np.float32).reshape(6, 2, 2, 2) / 17
b_data = np.linspace(-0.2, 0.2, 6, dtype=np.float32)

x = bg.tensor(x_data)
w = bg.tensor(w_data)
b = bg.tensor(b_data)

def loss_x(inp):
    return F.conv2d(inp, w, b, stride=(1,1), padding=(1,0), dilation=(1,1), groups=2).sum()

def loss_w(weight):
    return F.conv2d(x, weight, b, stride=(1,1), padding=(1,0), dilation=(1,1), groups=2).sum()

def loss_b(bias):
    return F.conv2d(x, w, bias, stride=(1,1), padding=(1,0), dilation=(1,1), groups=2).sum()

gx = bg.func.grad(loss_x)(x)
gw = bg.func.grad(loss_w)(w)
gb = bg.func.grad(loss_b)(b)

x_ref = bg.tensor(x_data, requires_grad=True)
w_ref = bg.tensor(w_data, requires_grad=True)
b_ref = bg.tensor(b_data, requires_grad=True)
F.conv2d(x_ref, w_ref, b_ref, stride=(1,1), padding=(1,0), dilation=(1,1), groups=2).sum().backward()

{
    "gxClose": bool(np.allclose(gx.numpy(), x_ref.grad.numpy(), atol=1e-5)),
    "gwClose": bool(np.allclose(gw.numpy(), w_ref.grad.numpy(), atol=1e-5)),
    "gbClose": bool(np.allclose(gb.numpy(), b_ref.grad.numpy(), atol=1e-5)),
    "gxOps": [u.op for u in toposort(gx._uop)],
    "gwOps": [u.op for u in toposort(gw._uop)],
    "gbOps": [u.op for u in toposort(gb._uop)],
    "registered": list(list_registered()),
}
`);
    expect(result.gxClose).toBe(true);
    expect(result.gwClose).toBe(true);
    expect(result.gbClose).toBe(true);
    expect(result.gxOps).toContain("CONV2D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV2D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV2D_BACKWARD_BIAS");
    expect(result.registered).toContain("CONV2D");
  });

  it("conv_transpose2d supports groups, output_padding, dilation, and backward", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      forwardClose: boolean;
      outShape: number[];
      op: string;
      gradShapes: number[][];
      gradNonzero: boolean[];
      gxClose: boolean;
      gwClose: boolean;
      gbClose: boolean;
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
      registered: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
import browsergrad_jit.nn.functional as F
from browsergrad_jit._ir import toposort
from browsergrad_jit._vjp import list_registered

def conv_transpose2d_ref(x, w, b, stride=(1,1), padding=(0,0), output_padding=(0,0), groups=1, dilation=(1,1)):
    N, C_in, H, W = x.shape
    _, copg, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    oph, opw = output_padding
    dh, dw = dilation
    C_out = copg * groups
    H_out = (H - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    W_out = (W - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    in_per_group = C_in // groups
    for n in range(N):
        for ci in range(C_in):
            g = ci // in_per_group
            co0 = g * copg
            for ih in range(H):
                for iw in range(W):
                    for r in range(kh):
                        oh = ih * sh - ph + r * dh
                        if oh < 0 or oh >= H_out:
                            continue
                        for c in range(kw):
                            ow = iw * sw - pw + c * dw
                            if 0 <= ow < W_out:
                                out[n, co0:co0+copg, oh, ow] += x[n, ci, ih, iw] * w[ci, :, r, c]
    return out + b.reshape(1, C_out, 1, 1)

x_data = np.arange(1, 1 + 1*4*2*3, dtype=np.float32).reshape(1, 4, 2, 3) / 10
w_data = np.arange(1, 1 + 4*2*2*2, dtype=np.float32).reshape(4, 2, 2, 2) / 30
b_data = np.linspace(-0.2, 0.2, 4, dtype=np.float32)
x = bg.tensor(x_data, requires_grad=True)
w = bg.tensor(w_data, requires_grad=True)
b = bg.tensor(b_data, requires_grad=True)
y = F.conv_transpose2d(x, w, b, stride=(2,1), padding=(1,0), output_padding=(1,0), groups=2, dilation=(2,1))
expected = conv_transpose2d_ref(x_data, w_data, b_data, stride=(2,1), padding=(1,0), output_padding=(1,0), groups=2, dilation=(2,1))
y.sum().backward()

def loss_x(inp):
    return F.conv_transpose2d(inp, w, b, stride=(2,1), padding=(1,0), output_padding=(1,0), groups=2, dilation=(2,1)).sum()

def loss_w(weight):
    return F.conv_transpose2d(x, weight, b, stride=(2,1), padding=(1,0), output_padding=(1,0), groups=2, dilation=(2,1)).sum()

def loss_b(bias):
    return F.conv_transpose2d(x, w, bias, stride=(2,1), padding=(1,0), output_padding=(1,0), groups=2, dilation=(2,1)).sum()

gx = bg.func.grad(loss_x)(x)
gw = bg.func.grad(loss_w)(w)
gb = bg.func.grad(loss_b)(b)
{
    "forwardClose": bool(np.allclose(y.numpy(), expected, atol=1e-5)),
    "outShape": list(y.shape),
    "op": y._uop.op,
    "gradShapes": [list(x.grad.shape), list(w.grad.shape), list(b.grad.shape)],
    "gradNonzero": [
        bool(np.abs(x.grad.numpy()).sum() > 0),
        bool(np.abs(w.grad.numpy()).sum() > 0),
        bool(np.abs(b.grad.numpy()).sum() > 0),
    ],
    "gxClose": bool(np.allclose(gx.numpy(), x.grad.numpy(), atol=1e-5)),
    "gwClose": bool(np.allclose(gw.numpy(), w.grad.numpy(), atol=1e-5)),
    "gbClose": bool(np.allclose(gb.numpy(), b.grad.numpy(), atol=1e-5)),
    "gxOps": [u.op for u in toposort(gx._uop)],
    "gwOps": [u.op for u in toposort(gw._uop)],
    "gbOps": [u.op for u in toposort(gb._uop)],
    "registered": list(list_registered()),
}
`);
    expect(result.forwardClose).toBe(true);
    expect(result.outShape).toEqual([1, 4, 4, 4]);
    expect(result.op).toBe("CONV_TRANSPOSE2D");
    expect(result.gradShapes).toEqual([[1, 4, 2, 3], [4, 2, 2, 2], [4]]);
    expect(result.gradNonzero).toEqual([true, true, true]);
    expect(result.gxClose).toBe(true);
    expect(result.gwClose).toBe(true);
    expect(result.gbClose).toBe(true);
    expect(result.gxOps).toContain("CONV_TRANSPOSE2D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV_TRANSPOSE2D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV_TRANSPOSE2D_BACKWARD_BIAS");
    expect(result.registered).toContain("CONV_TRANSPOSE2D");
  });

  it("conv3d supports groups, dilation, tuple shapes, and backward", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      forwardClose: boolean;
      outShape: number[];
      op: string;
      gradShapes: number[][];
      gradNonzero: boolean[];
      gxClose: boolean;
      gwClose: boolean;
      gbClose: boolean;
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
      registered: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
import browsergrad_jit.nn.functional as F
from browsergrad_jit._ir import toposort
from browsergrad_jit._vjp import list_registered

def conv3d_ref(x, w, b, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1):
    N, C_in, D, H, W = x.shape
    C_out, cpg, kd, kh, kw = w.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    x_pad = np.pad(x, ((0,0),(0,0),(pd,pd),(ph,ph),(pw,pw)), mode="constant")
    D_out = (D + 2*pd - (dd*(kd-1)+1)) // sd + 1
    H_out = (H + 2*ph - (dh*(kh-1)+1)) // sh + 1
    W_out = (W + 2*pw - (dw*(kw-1)+1)) // sw + 1
    out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=np.float32)
    opg = C_out // groups
    for n in range(N):
        for g in range(groups):
            c0 = g * cpg
            o0 = g * opg
            for co in range(opg):
                for od in range(D_out):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            d0, h0, w0 = od*sd, oh*sh, ow*sw
                            out[n, o0+co, od, oh, ow] = (
                                x_pad[n, c0:c0+cpg, d0:d0+dd*(kd-1)+1:dd, h0:h0+dh*(kh-1)+1:dh, w0:w0+dw*(kw-1)+1:dw]
                                * w[o0+co]
                            ).sum()
    return out + b.reshape(1, C_out, 1, 1, 1)

x_data = np.arange(1, 1 + 1*4*3*4*3, dtype=np.float32).reshape(1, 4, 3, 4, 3) / 20
w_data = np.arange(1, 1 + 6*2*2*2*2, dtype=np.float32).reshape(6, 2, 2, 2, 2) / 40
b_data = np.linspace(-0.1, 0.1, 6, dtype=np.float32)
x = bg.tensor(x_data, requires_grad=True)
w = bg.tensor(w_data, requires_grad=True)
b = bg.tensor(b_data, requires_grad=True)
y = F.conv3d(x, w, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2)
expected = conv3d_ref(x_data, w_data, b_data, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2)
y.sum().backward()
gx = bg.func.grad(lambda inp: F.conv3d(inp, w, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(x)
gw = bg.func.grad(lambda weight: F.conv3d(x, weight, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(w)
gb = bg.func.grad(lambda bias: F.conv3d(x, w, bias, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(b)
{
    "forwardClose": bool(np.allclose(y.numpy(), expected, atol=1e-5)),
    "outShape": list(y.shape),
    "op": y._uop.op,
    "gradShapes": [list(x.grad.shape), list(w.grad.shape), list(b.grad.shape)],
    "gradNonzero": [
        bool(np.abs(x.grad.numpy()).sum() > 0),
        bool(np.abs(w.grad.numpy()).sum() > 0),
        bool(np.abs(b.grad.numpy()).sum() > 0),
    ],
    "gxClose": bool(np.allclose(gx.numpy(), x.grad.numpy(), atol=1e-5)),
    "gwClose": bool(np.allclose(gw.numpy(), w.grad.numpy(), atol=1e-5)),
    "gbClose": bool(np.allclose(gb.numpy(), b.grad.numpy(), atol=1e-5)),
    "gxOps": [u.op for u in toposort(gx._uop)],
    "gwOps": [u.op for u in toposort(gw._uop)],
    "gbOps": [u.op for u in toposort(gb._uop)],
    "registered": list(list_registered()),
}
`);
    expect(result.forwardClose).toBe(true);
    expect(result.outShape).toEqual([1, 6, 4, 4, 2]);
    expect(result.op).toBe("CONV3D");
    expect(result.gradShapes).toEqual([[1, 4, 3, 4, 3], [6, 2, 2, 2, 2], [6]]);
    expect(result.gradNonzero).toEqual([true, true, true]);
    expect(result.gxClose).toBe(true);
    expect(result.gwClose).toBe(true);
    expect(result.gbClose).toBe(true);
    expect(result.gxOps).toContain("CONV3D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV3D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV3D_BACKWARD_BIAS");
    expect(result.registered).toContain("CONV3D");
  });
});
