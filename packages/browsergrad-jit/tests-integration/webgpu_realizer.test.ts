/**
 * WebGPU realizer seam tests (PRD-011.5 spike).
 *
 * The real bridge dispatches WGSL kernels via Pyodide → JS → WebGPU.
 * That path needs a real GPUDevice and lives in browser CI. These
 * tests instead instantiate a Python-side NumPy-backed mock bridge
 * that satisfies the same surface (see `_bridge.py` Protocol) and
 * verify:
 *
 *   - The seam dispatches correctly (every supported UOp routes
 *     to the bridge method and the returned handle threads through).
 *   - Residency contract: chained matmuls trigger N uploads + 1
 *     materialise, with intermediate handles released the moment
 *     their last consumer finishes.
 *   - End-to-end numerical match between bg.realize_webgpu and
 *     bg.tensor.numpy() (which uses the NumPy realizer).
 *   - Refusal modes: unsupported opcode raises with a pointer to
 *     bg.realize() as the fallback.
 *   - Flash Attention opt-in CUSTOM op routes through bridge.flash_attention.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

const MOCK_BRIDGE_PY = `
# A NumPy-backed mock bridge that satisfies the WebGpuBridge protocol.
# Used for testing the seam in pyodide-in-node where no GPUDevice exists.
#
# Records every call so tests can assert on the residency contract:
#   - upload_count / materialize_count / release_count
#   - alive set: handles minted minus handles released
import numpy as np

class MockBridge:
    def __init__(self):
        self._handles = {}        # handle_id -> ndarray
        self._next = 0
        self.upload_count = 0
        self.materialize_count = 0
        self.release_count = 0
        self.matmul_count = 0
        self.fused_count = 0
        self.flash_count = 0
        self.conv1d_count = 0
        self.conv1d_backward_input_count = 0
        self.conv1d_backward_weight_count = 0
        self.conv1d_backward_bias_count = 0
        self.conv2d_count = 0
        self.conv2d_backward_input_count = 0
        self.conv2d_backward_weight_count = 0
        self.conv2d_backward_bias_count = 0
        self.cast_count = 0
        self.tensor_plan_count = 0
        self.tensor_plan_resident_count = 0
        self.calls = []           # ordered list of (op, handle_id_in_or_out)

    def _mint(self, arr):
        hid = self._next
        self._next += 1
        self._handles[hid] = np.array(arr, copy=True)
        return hid

    def alive(self):
        return set(self._handles.keys())

    # ---- protocol ----
    def upload(self, data, shape, dtype):
        self.upload_count += 1
        arr = np.frombuffer(data, dtype=np.dtype(dtype))
        if shape and shape != (1,):
            arr = arr.reshape(shape)
        h = self._mint(arr)
        self.calls.append(("upload", h))
        return h

    def materialize(self, handle, shape, dtype):
        self.materialize_count += 1
        arr = self._handles[handle]
        self.calls.append(("materialize", handle))
        return arr.astype(np.dtype(dtype), copy=False).tobytes()

    def release(self, handle):
        self.release_count += 1
        self.calls.append(("release", handle))
        self._handles.pop(handle, None)

    def matmul(self, a, b, m, k, n, dtype):
        self.matmul_count += 1
        out = self._handles[a] @ self._handles[b]
        h = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("matmul", h))
        return h

    def fused_elementwise(self, inputs, ops, shape, dtype):
        self.fused_count += 1
        # ops is a list of (opcode, lhs_ref, rhs_ref) — same shape as the
        # NumPy realizer's _h_fused_elementwise.
        externals = [self._handles[h] for h in inputs]
        steps = []
        def resolve(ref):
            return externals[-ref - 1] if ref < 0 else steps[ref]
        for opcode, lhs_ref, rhs_ref in ops:
            a = resolve(lhs_ref)
            if opcode == "ADD":
                steps.append(a + resolve(rhs_ref))
            elif opcode == "MUL":
                steps.append(a * resolve(rhs_ref))
            elif opcode == "DIV":
                steps.append(a / resolve(rhs_ref))
            elif opcode == "NEG":
                steps.append(-a)
            elif opcode == "EXP":
                steps.append(np.exp(a))
            elif opcode == "LOG":
                steps.append(np.log(a))
            else:
                raise ValueError(f"mock fused: unknown op {opcode}")
        h = self._mint(steps[-1].astype(np.dtype(dtype), copy=False))
        self.calls.append(("fused_elementwise", h))
        return h

    def cast(self, handle, src_dtype, dst_dtype, shape):
        self.cast_count += 1
        arr = self._handles[handle].astype(np.dtype(dst_dtype), copy=False)
        h = self._mint(arr)
        self.calls.append(("cast", h))
        return h

    def flash_attention(self, q, k, v, mask, b, h_, sq, sk, d, scale, dtype):
        self.flash_count += 1
        Q = self._handles[q]
        K = self._handles[k]
        V = self._handles[v]
        # Composed reference: scores = Q @ K^T * scale; (+ mask); softmax; @ V.
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
        if mask is not None:
            scores = scores + self._handles[mask]
        # Stable softmax along the last axis.
        m_ = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m_)
        p = e / e.sum(axis=-1, keepdims=True)
        out = np.matmul(p, V).astype(np.dtype(dtype), copy=False)
        hh = self._mint(out)
        self.calls.append(("flash_attention", hh))
        return hh

    def conv1d(self, input, weight, bias, n, c_in, l_in, c_out, k, stride, padding, dilation, groups, l_out, dtype):
        self.conv1d_count += 1
        x = self._handles[input]
        wt = self._handles[weight]
        b_arr = self._handles[bias] if bias is not None else None
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding)), mode="constant")
        out = np.zeros((n, c_out, l_out), dtype=np.float32)
        cpg = c_in // groups
        opg = c_out // groups
        eff_k = dilation * (k - 1) + 1
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    for i in range(l_out):
                        l0 = i * stride
                        out[nn, o0 + co, i] = (
                            x_pad[nn, c0:c0+cpg, l0:l0+eff_k:dilation]
                            * wt[o0 + co]
                        ).sum()
        if b_arr is not None:
            out = out + b_arr.reshape(1, c_out, 1)
        hh = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv1d", hh))
        return hh

    def conv1d_backward_input(self, dy, weight, n, c_in, l_in, c_out, k, stride, padding, dilation, groups, l_out, dtype):
        self.conv1d_backward_input_count += 1
        grad_out = self._handles[dy]
        wt = self._handles[weight]
        cpg = c_in // groups
        opg = c_out // groups
        grad_x = np.zeros((n, c_in, l_in + 2 * padding), dtype=np.float32)
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    out_ch = o0 + co
                    for i in range(l_out):
                        grad_val = grad_out[nn, out_ch, i]
                        base = i * stride
                        for ci_local in range(cpg):
                            in_ch = c0 + ci_local
                            for r in range(k):
                                li = base + r * dilation
                                grad_x[nn, in_ch, li] += grad_val * wt[out_ch, ci_local, r]
        out = grad_x[:, :, padding:padding+l_in] if padding > 0 else grad_x
        hh = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv1d_backward_input", hh))
        return hh

    def conv1d_backward_weight(self, dy, input, n, c_in, l_in, c_out, k, stride, padding, dilation, groups, l_out, dtype):
        self.conv1d_backward_weight_count += 1
        grad_out = self._handles[dy]
        x = self._handles[input]
        cpg = c_in // groups
        opg = c_out // groups
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding)), mode="constant")
        grad_w = np.zeros((c_out, cpg, k), dtype=np.float32)
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    out_ch = o0 + co
                    for ci_local in range(cpg):
                        in_ch = c0 + ci_local
                        for r in range(k):
                            acc = 0.0
                            for i in range(l_out):
                                li = i * stride + r * dilation
                                acc += grad_out[nn, out_ch, i] * x_pad[nn, in_ch, li]
                            grad_w[out_ch, ci_local, r] += acc
        hh = self._mint(grad_w.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv1d_backward_weight", hh))
        return hh

    def conv1d_backward_bias(self, dy, n, c_out, l_out, dtype):
        self.conv1d_backward_bias_count += 1
        out = self._handles[dy].sum(axis=(0, 2)).astype(np.dtype(dtype), copy=False)
        hh = self._mint(out)
        self.calls.append(("conv1d_backward_bias", hh))
        return hh

    def conv2d(self, input, weight, bias, n, c_in, h, w, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, out_h, out_w, dtype):
        self.conv2d_count += 1
        x = self._handles[input]
        wt = self._handles[weight]
        b_arr = self._handles[bias] if bias is not None else None
        x_pad = np.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
        out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
        cpg = c_in // groups
        opg = c_out // groups
        eff_h = dilation_h * (kh - 1) + 1
        eff_w = dilation_w * (kw - 1) + 1
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            h0 = oh * stride_h
                            w0 = ow * stride_w
                            out[nn, o0 + co, oh, ow] = (
                                x_pad[nn, c0:c0+cpg, h0:h0+eff_h:dilation_h, w0:w0+eff_w:dilation_w]
                                * wt[o0 + co]
                            ).sum()
        if b_arr is not None:
            out = out + b_arr.reshape(1, c_out, 1, 1)
        hh = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv2d", hh))
        return hh

    def conv2d_backward_input(self, dy, weight, n, c_in, h, w, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, out_h, out_w, dtype):
        self.conv2d_backward_input_count += 1
        grad_out = self._handles[dy]
        wt = self._handles[weight]
        cpg = c_in // groups
        opg = c_out // groups
        grad_x = np.zeros((n, c_in, h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    out_ch = o0 + co
                    for oh in range(out_h):
                        for ow in range(out_w):
                            grad_val = grad_out[nn, out_ch, oh, ow]
                            h_base = oh * stride_h
                            w_base = ow * stride_w
                            for ci_local in range(cpg):
                                in_ch = c0 + ci_local
                                for r in range(kh):
                                    ih = h_base + r * dilation_h
                                    for s in range(kw):
                                        iw = w_base + s * dilation_w
                                        grad_x[nn, in_ch, ih, iw] += grad_val * wt[out_ch, ci_local, r, s]
        out = grad_x[:, :, pad_h:pad_h+h, pad_w:pad_w+w] if (pad_h > 0 or pad_w > 0) else grad_x
        hh = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv2d_backward_input", hh))
        return hh

    def conv2d_backward_weight(self, dy, input, n, c_in, h, w, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, out_h, out_w, dtype):
        self.conv2d_backward_weight_count += 1
        grad_out = self._handles[dy]
        x = self._handles[input]
        cpg = c_in // groups
        opg = c_out // groups
        x_pad = np.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
        grad_w = np.zeros((c_out, cpg, kh, kw), dtype=np.float32)
        for nn in range(n):
            for g in range(groups):
                c0 = g * cpg
                o0 = g * opg
                for co in range(opg):
                    out_ch = o0 + co
                    for ci_local in range(cpg):
                        in_ch = c0 + ci_local
                        for r in range(kh):
                            for s in range(kw):
                                acc = 0.0
                                for oh in range(out_h):
                                    ih = oh * stride_h + r * dilation_h
                                    for ow in range(out_w):
                                        iw = ow * stride_w + s * dilation_w
                                        acc += grad_out[nn, out_ch, oh, ow] * x_pad[nn, in_ch, ih, iw]
                                grad_w[out_ch, ci_local, r, s] += acc
        hh = self._mint(grad_w.astype(np.dtype(dtype), copy=False))
        self.calls.append(("conv2d_backward_weight", hh))
        return hh

    def conv2d_backward_bias(self, dy, n, c_out, out_h, out_w, dtype):
        self.conv2d_backward_bias_count += 1
        out = self._handles[dy].sum(axis=(0, 2, 3)).astype(np.dtype(dtype), copy=False)
        hh = self._mint(out)
        self.calls.append(("conv2d_backward_bias", hh))
        return hh

    def run_tensor_plan(self, plan, inputs, dtype):
        self.tensor_plan_count += 1
        if plan.get("has_custom_ops"):
            raise ValueError("mock tensor plan refuses CUSTOM-backed plans")
        provided = {}
        for item in inputs:
            value_id = item.get("value_id", item.get("valueId"))
            if "handle" in item:
                provided[value_id] = self._handles[item["handle"]].reshape(-1)
            else:
                data = item["data"]
                provided[value_id] = np.frombuffer(data, dtype=np.dtype(dtype))
        values = {}
        for step in plan["steps"]:
            value_id = step.get("value_id", step.get("valueId"))
            input_ids = step.get("input_ids", step.get("inputIds", []))
            shape = tuple(step["shape"])
            op = step["op"]
            if op == "BUFFER":
                arr = provided[value_id]
                values[value_id] = arr.reshape(shape) if shape else arr.reshape(())
            elif op == "LOAD":
                values[value_id] = values[input_ids[0]]
            elif op == "MATMUL":
                values[value_id] = values[input_ids[0]] @ values[input_ids[1]]
            elif op == "ADD":
                values[value_id] = values[input_ids[0]] + values[input_ids[1]]
            elif op == "MUL":
                values[value_id] = values[input_ids[0]] * values[input_ids[1]]
            elif op == "DIV":
                values[value_id] = values[input_ids[0]] / values[input_ids[1]]
            elif op == "NEG":
                values[value_id] = -values[input_ids[0]]
            elif op == "EXP":
                values[value_id] = np.exp(values[input_ids[0]])
            elif op == "LOG":
                values[value_id] = np.log(values[input_ids[0]])
            elif op == "FUSED_ELEMENTWISE":
                externals = [values[input_id] for input_id in input_ids]
                steps = []
                def resolve(ref):
                    return externals[-ref - 1] if ref < 0 else steps[ref]
                for opcode, lhs_ref, rhs_ref in step["arg"]["ops"]:
                    a = resolve(int(lhs_ref))
                    rhs = int(rhs_ref) if rhs_ref is not None else int(lhs_ref)
                    if opcode == "ADD":
                        steps.append(a + resolve(rhs))
                    elif opcode == "MUL":
                        steps.append(a * resolve(rhs))
                    elif opcode == "DIV":
                        steps.append(a / resolve(rhs))
                    elif opcode == "NEG":
                        steps.append(-a)
                    elif opcode == "EXP":
                        steps.append(np.exp(a))
                    elif opcode == "LOG":
                        steps.append(np.log(a))
                    else:
                        raise ValueError(f"mock tensor plan: unsupported fused opcode {opcode}")
                values[value_id] = steps[-1].astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CAST":
                values[value_id] = values[input_ids[0]].astype(np.dtype(step["dtype"]), copy=False)
            elif op == "RESHAPE":
                values[value_id] = values[input_ids[0]].reshape(shape)
            elif op == "PERMUTE":
                values[value_id] = np.transpose(values[input_ids[0]], axes=tuple(step["arg"]["axes"]))
            elif op == "BROADCAST_TO":
                values[value_id] = np.broadcast_to(values[input_ids[0]], shape).copy()
            elif op == "REDUCE":
                reduce_op = step["arg"]["op"]
                axis = step["arg"].get("axis", None)
                keepdims = bool(step["arg"].get("keepdims", False))
                if isinstance(axis, list):
                    axis = tuple(axis)
                if reduce_op == "sum":
                    values[value_id] = np.sum(values[input_ids[0]], axis=axis, keepdims=keepdims)
                elif reduce_op == "mean":
                    values[value_id] = np.mean(values[input_ids[0]], axis=axis, keepdims=keepdims)
                else:
                    raise ValueError(f"mock tensor plan: unsupported reduce {reduce_op}")
            elif op == "CONV1D":
                arg = step["arg"]
                x = values[input_ids[0]]
                wt = values[input_ids[1]]
                b_arr = values[input_ids[2]] if len(input_ids) > 2 else None
                n = int(arg["n"]); c_in = int(arg["c_in"]); l_in = int(arg["l_in"])
                c_out = int(arg["c_out"]); k = int(arg["k"])
                stride = int(arg["stride"]); padding = int(arg["padding"])
                dilation = int(arg["dilation"]); groups = int(arg["groups"])
                l_out = int(arg["l_out"])
                x_pad = np.pad(x, ((0,0),(0,0),(padding,padding)), mode="constant")
                out = np.zeros((n, c_out, l_out), dtype=np.float32)
                cpg = c_in // groups
                opg = c_out // groups
                eff_k = dilation * (k - 1) + 1
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            for i in range(l_out):
                                l0 = i * stride
                                out[nn, o0 + co, i] = (
                                    x_pad[nn, c0:c0+cpg, l0:l0+eff_k:dilation]
                                    * wt[o0 + co]
                                ).sum()
                if b_arr is not None:
                    out = out + b_arr.reshape(1, c_out, 1)
                values[value_id] = out.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV1D_BACKWARD_INPUT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                wt = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"]); l_in = int(arg["l_in"])
                c_out = int(arg["c_out"]); k = int(arg["k"])
                stride = int(arg["stride"]); padding = int(arg["padding"])
                dilation = int(arg["dilation"]); groups = int(arg["groups"])
                l_out = int(arg["l_out"])
                cpg = c_in // groups
                opg = c_out // groups
                grad_x = np.zeros((n, c_in, l_in + 2 * padding), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for i in range(l_out):
                                grad_val = grad_out[nn, out_ch, i]
                                base = i * stride
                                for ci_local in range(cpg):
                                    in_ch = c0 + ci_local
                                    for r in range(k):
                                        li = base + r * dilation
                                        grad_x[nn, in_ch, li] += grad_val * wt[out_ch, ci_local, r]
                values[value_id] = (
                    grad_x[:, :, padding:padding+l_in] if padding > 0 else grad_x
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV1D_BACKWARD_WEIGHT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                x = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"]); l_in = int(arg["l_in"])
                c_out = int(arg["c_out"]); k = int(arg["k"])
                stride = int(arg["stride"]); padding = int(arg["padding"])
                dilation = int(arg["dilation"]); groups = int(arg["groups"])
                l_out = int(arg["l_out"])
                cpg = c_in // groups
                opg = c_out // groups
                x_pad = np.pad(x, ((0,0),(0,0),(padding,padding)), mode="constant")
                grad_w = np.zeros((c_out, cpg, k), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for ci_local in range(cpg):
                                in_ch = c0 + ci_local
                                for r in range(k):
                                    acc = 0.0
                                    for i in range(l_out):
                                        li = i * stride + r * dilation
                                        acc += grad_out[nn, out_ch, i] * x_pad[nn, in_ch, li]
                                    grad_w[out_ch, ci_local, r] += acc
                values[value_id] = grad_w.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV1D_BACKWARD_BIAS":
                values[value_id] = values[input_ids[0]].sum(axis=(0, 2)).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV2D":
                arg = step["arg"]
                x = values[input_ids[0]]
                wt = values[input_ids[1]]
                b_arr = values[input_ids[2]] if len(input_ids) > 2 else None
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                x_pad = np.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
                out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
                cpg = c_in // groups
                opg = c_out // groups
                eff_h = dilation_h * (kh - 1) + 1
                eff_w = dilation_w * (kw - 1) + 1
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            for oh in range(out_h):
                                for ow in range(out_w):
                                    h0 = oh * stride_h
                                    w0 = ow * stride_w
                                    out[nn, o0 + co, oh, ow] = (
                                        x_pad[nn, c0:c0+cpg, h0:h0+eff_h:dilation_h, w0:w0+eff_w:dilation_w]
                                        * wt[o0 + co]
                                    ).sum()
                if b_arr is not None:
                    out = out + b_arr.reshape(1, c_out, 1, 1)
                values[value_id] = out.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV2D_BACKWARD_INPUT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                wt = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                cpg = c_in // groups
                opg = c_out // groups
                grad_x = np.zeros((n, c_in, h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for oh in range(out_h):
                                for ow in range(out_w):
                                    grad_val = grad_out[nn, out_ch, oh, ow]
                                    h_base = oh * stride_h
                                    w_base = ow * stride_w
                                    for ci_local in range(cpg):
                                        in_ch = c0 + ci_local
                                        for r in range(kh):
                                            ih = h_base + r * dilation_h
                                            for s in range(kw):
                                                iw = w_base + s * dilation_w
                                                grad_x[nn, in_ch, ih, iw] += grad_val * wt[out_ch, ci_local, r, s]
                values[value_id] = (
                    grad_x[:, :, pad_h:pad_h+h, pad_w:pad_w+w]
                    if (pad_h > 0 or pad_w > 0)
                    else grad_x
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV2D_BACKWARD_WEIGHT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                x = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                cpg = c_in // groups
                opg = c_out // groups
                x_pad = np.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
                grad_w = np.zeros((c_out, cpg, kh, kw), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for ci_local in range(cpg):
                                in_ch = c0 + ci_local
                                for r in range(kh):
                                    for s in range(kw):
                                        acc = 0.0
                                        for oh in range(out_h):
                                            ih = oh * stride_h + r * dilation_h
                                            for ow in range(out_w):
                                                iw = ow * stride_w + s * dilation_w
                                                acc += grad_out[nn, out_ch, oh, ow] * x_pad[nn, in_ch, ih, iw]
                                        grad_w[out_ch, ci_local, r, s] += acc
                values[value_id] = grad_w.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV2D_BACKWARD_BIAS":
                values[value_id] = values[input_ids[0]].sum(axis=(0, 2, 3)).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV_TRANSPOSE2D":
                arg = step["arg"]
                x = values[input_ids[0]]
                wt = values[input_ids[1]]
                b_arr = values[input_ids[2]] if len(input_ids) > 2 else None
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                c_out_per_group = int(arg["c_out_per_group"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                in_per_group = c_in // groups
                out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
                for nn in range(n):
                    for ci in range(c_in):
                        group = ci // in_per_group
                        co0 = group * c_out_per_group
                        for ih in range(h):
                            for iw in range(w):
                                x_val = x[nn, ci, ih, iw]
                                for r in range(kh):
                                    oh = ih * stride_h - pad_h + r * dilation_h
                                    if oh < 0 or oh >= out_h:
                                        continue
                                    for s in range(kw):
                                        ow = iw * stride_w - pad_w + s * dilation_w
                                        if 0 <= ow < out_w:
                                            out[nn, co0:co0+c_out_per_group, oh, ow] += x_val * wt[ci, :, r, s]
                if b_arr is not None:
                    out = out + b_arr.reshape(1, c_out, 1, 1)
                values[value_id] = out.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV_TRANSPOSE2D_BACKWARD_INPUT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                wt = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"])
                c_out_per_group = int(arg["c_out_per_group"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                in_per_group = c_in // groups
                grad_x = np.zeros((n, c_in, h, w), dtype=np.float32)
                for nn in range(n):
                    for ci in range(c_in):
                        group = ci // in_per_group
                        co0 = group * c_out_per_group
                        for ih in range(h):
                            for iw in range(w):
                                acc = 0.0
                                for r in range(kh):
                                    oh = ih * stride_h - pad_h + r * dilation_h
                                    if oh < 0 or oh >= out_h:
                                        continue
                                    for s in range(kw):
                                        ow = iw * stride_w - pad_w + s * dilation_w
                                        if 0 <= ow < out_w:
                                            acc += (grad_out[nn, co0:co0+c_out_per_group, oh, ow] * wt[ci, :, r, s]).sum()
                                grad_x[nn, ci, ih, iw] = acc
                values[value_id] = grad_x.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV_TRANSPOSE2D_BACKWARD_WEIGHT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                x = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                h = int(arg["h"]); w = int(arg["w"])
                c_out_per_group = int(arg["c_out_per_group"])
                kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                in_per_group = c_in // groups
                grad_w = np.zeros((c_in, c_out_per_group, kh, kw), dtype=np.float32)
                for nn in range(n):
                    for ci in range(c_in):
                        group = ci // in_per_group
                        co0 = group * c_out_per_group
                        for ih in range(h):
                            for iw in range(w):
                                x_val = x[nn, ci, ih, iw]
                                for r in range(kh):
                                    oh = ih * stride_h - pad_h + r * dilation_h
                                    if oh < 0 or oh >= out_h:
                                        continue
                                    for s in range(kw):
                                        ow = iw * stride_w - pad_w + s * dilation_w
                                        if 0 <= ow < out_w:
                                            grad_w[ci, :, r, s] += grad_out[nn, co0:co0+c_out_per_group, oh, ow] * x_val
                values[value_id] = grad_w.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV_TRANSPOSE2D_BACKWARD_BIAS":
                values[value_id] = values[input_ids[0]].sum(axis=(0, 2, 3)).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV3D":
                arg = step["arg"]
                x = values[input_ids[0]]
                wt = values[input_ids[1]]
                b_arr = values[input_ids[2]] if len(input_ids) > 2 else None
                n = int(arg["n"]); c_in = int(arg["c_in"])
                d = int(arg["d"]); h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kd = int(arg["kd"]); kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_d = int(arg["stride_d"]); stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_d = int(arg["pad_d"]); pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_d = int(arg["dilation_d"]); dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_d = int(arg["out_d"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                x_pad = np.pad(x, ((0,0),(0,0),(pad_d,pad_d),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
                out = np.zeros((n, c_out, out_d, out_h, out_w), dtype=np.float32)
                cpg = c_in // groups
                opg = c_out // groups
                eff_d = dilation_d * (kd - 1) + 1
                eff_h = dilation_h * (kh - 1) + 1
                eff_w = dilation_w * (kw - 1) + 1
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            for od in range(out_d):
                                for oh in range(out_h):
                                    for ow in range(out_w):
                                        d0 = od * stride_d
                                        h0 = oh * stride_h
                                        w0 = ow * stride_w
                                        out[nn, o0 + co, od, oh, ow] = (
                                            x_pad[nn, c0:c0+cpg, d0:d0+eff_d:dilation_d, h0:h0+eff_h:dilation_h, w0:w0+eff_w:dilation_w]
                                            * wt[o0 + co]
                                        ).sum()
                if b_arr is not None:
                    out = out + b_arr.reshape(1, c_out, 1, 1, 1)
                values[value_id] = out.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV3D_BACKWARD_INPUT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                wt = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                d = int(arg["d"]); h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kd = int(arg["kd"]); kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_d = int(arg["stride_d"]); stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_d = int(arg["pad_d"]); pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_d = int(arg["dilation_d"]); dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_d = int(arg["out_d"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                cpg = c_in // groups
                opg = c_out // groups
                grad_x = np.zeros((n, c_in, d + 2 * pad_d, h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for od in range(out_d):
                                for oh in range(out_h):
                                    for ow in range(out_w):
                                        grad_val = grad_out[nn, out_ch, od, oh, ow]
                                        d_base = od * stride_d
                                        h_base = oh * stride_h
                                        w_base = ow * stride_w
                                        for ci_local in range(cpg):
                                            in_ch = c0 + ci_local
                                            for rd in range(kd):
                                                di = d_base + rd * dilation_d
                                                for rh in range(kh):
                                                    hi = h_base + rh * dilation_h
                                                    for rw in range(kw):
                                                        wi = w_base + rw * dilation_w
                                                        grad_x[nn, in_ch, di, hi, wi] += grad_val * wt[out_ch, ci_local, rd, rh, rw]
                values[value_id] = (
                    grad_x[:, :, pad_d:pad_d+d, pad_h:pad_h+h, pad_w:pad_w+w]
                    if (pad_d > 0 or pad_h > 0 or pad_w > 0)
                    else grad_x
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV3D_BACKWARD_WEIGHT":
                arg = step["arg"]
                grad_out = values[input_ids[0]]
                x = values[input_ids[1]]
                n = int(arg["n"]); c_in = int(arg["c_in"])
                d = int(arg["d"]); h = int(arg["h"]); w = int(arg["w"]); c_out = int(arg["c_out"])
                kd = int(arg["kd"]); kh = int(arg["kh"]); kw = int(arg["kw"])
                stride_d = int(arg["stride_d"]); stride_h = int(arg["stride_h"]); stride_w = int(arg["stride_w"])
                pad_d = int(arg["pad_d"]); pad_h = int(arg["pad_h"]); pad_w = int(arg["pad_w"])
                dilation_d = int(arg["dilation_d"]); dilation_h = int(arg["dilation_h"]); dilation_w = int(arg["dilation_w"])
                groups = int(arg["groups"]); out_d = int(arg["out_d"]); out_h = int(arg["out_h"]); out_w = int(arg["out_w"])
                cpg = c_in // groups
                opg = c_out // groups
                x_pad = np.pad(x, ((0,0),(0,0),(pad_d,pad_d),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")
                grad_w = np.zeros((c_out, cpg, kd, kh, kw), dtype=np.float32)
                for nn in range(n):
                    for g in range(groups):
                        c0 = g * cpg
                        o0 = g * opg
                        for co in range(opg):
                            out_ch = o0 + co
                            for ci_local in range(cpg):
                                in_ch = c0 + ci_local
                                for rd in range(kd):
                                    for rh in range(kh):
                                        for rw in range(kw):
                                            acc = 0.0
                                            for od in range(out_d):
                                                di = od * stride_d + rd * dilation_d
                                                for oh in range(out_h):
                                                    hi = oh * stride_h + rh * dilation_h
                                                    for ow in range(out_w):
                                                        wi = ow * stride_w + rw * dilation_w
                                                        acc += grad_out[nn, out_ch, od, oh, ow] * x_pad[nn, in_ch, di, hi, wi]
                                            grad_w[out_ch, ci_local, rd, rh, rw] += acc
                values[value_id] = grad_w.astype(np.dtype(step["dtype"]), copy=False)
            elif op == "CONV3D_BACKWARD_BIAS":
                values[value_id] = values[input_ids[0]].sum(axis=(0, 2, 3, 4)).astype(np.dtype(step["dtype"]), copy=False)
            elif op in ("LAYER_NORM", "LAYER_NORM_BACKWARD_INPUT", "LAYER_NORM_BACKWARD_WEIGHT", "LAYER_NORM_BACKWARD_BIAS"):
                arg = step["arg"]
                rows = int(arg["rows"]); cols = int(arg["cols"])
                eps = float(arg.get("eps", 1e-5))
                if op == "LAYER_NORM":
                    x = values[input_ids[0]]
                    weight = values[input_ids[1]].reshape(cols)
                    bias = values[input_ids[2]].reshape(cols)
                    x2 = x.reshape(rows, cols).astype(np.float32, copy=False)
                    mean = x2.mean(axis=1, keepdims=True)
                    centered = x2 - mean
                    inv_std = 1.0 / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + eps)
                    values[value_id] = (
                        centered * inv_std * weight.reshape(1, cols) + bias.reshape(1, cols)
                    ).reshape(x.shape).astype(np.dtype(step["dtype"]), copy=False)
                elif op == "LAYER_NORM_BACKWARD_INPUT":
                    dy = values[input_ids[0]]
                    x = values[input_ids[1]]
                    weight = values[input_ids[2]].reshape(cols)
                    x2 = x.reshape(rows, cols).astype(np.float32, copy=False)
                    dy2 = dy.reshape(rows, cols).astype(np.float32, copy=False)
                    mean = x2.mean(axis=1, keepdims=True)
                    centered = x2 - mean
                    inv_std = 1.0 / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + eps)
                    x_hat = centered * inv_std
                    g = dy2 * weight.reshape(1, cols)
                    sum_g = g.sum(axis=1, keepdims=True)
                    sum_g_xhat = (g * x_hat).sum(axis=1, keepdims=True)
                    dx = (inv_std / float(cols)) * (float(cols) * g - sum_g - x_hat * sum_g_xhat)
                    values[value_id] = dx.reshape(x.shape).astype(np.dtype(step["dtype"]), copy=False)
                elif op == "LAYER_NORM_BACKWARD_WEIGHT":
                    dy = values[input_ids[0]]
                    x = values[input_ids[1]]
                    x2 = x.reshape(rows, cols).astype(np.float32, copy=False)
                    dy2 = dy.reshape(rows, cols).astype(np.float32, copy=False)
                    mean = x2.mean(axis=1, keepdims=True)
                    centered = x2 - mean
                    inv_std = 1.0 / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + eps)
                    values[value_id] = (dy2 * centered * inv_std).sum(axis=0).reshape(step["shape"]).astype(np.dtype(step["dtype"]), copy=False)
                else:
                    dy = values[input_ids[0]]
                    values[value_id] = dy.reshape(rows, cols).sum(axis=0).reshape(step["shape"]).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "SGD_UPDATE":
                arg = step["arg"]
                param = values[input_ids[0]]
                grad = values[input_ids[1]]
                lr = float(arg["lr"])
                weight_decay = float(arg.get("weight_decay", 0.0))
                values[value_id] = (
                    param - lr * (grad + weight_decay * param)
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAMW_UPDATE_M":
                arg = step["arg"]
                m = values[input_ids[0]]
                grad = values[input_ids[1]]
                beta1 = float(arg["beta1"])
                values[value_id] = (
                    beta1 * m + (1.0 - beta1) * grad
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAMW_UPDATE_V":
                arg = step["arg"]
                v = values[input_ids[0]]
                grad = values[input_ids[1]]
                beta2 = float(arg["beta2"])
                values[value_id] = (
                    beta2 * v + (1.0 - beta2) * (grad * grad)
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAMW_UPDATE_PARAM":
                arg = step["arg"]
                param = values[input_ids[0]]
                m_new = values[input_ids[2]]
                v_new = values[input_ids[3]]
                lr = float(arg["lr"])
                beta1 = float(arg["beta1"])
                beta2 = float(arg["beta2"])
                eps = float(arg["eps"])
                step_i = int(arg["step"])
                weight_decay = float(arg.get("weight_decay", 0.0))
                m_hat = m_new / (1.0 - beta1 ** step_i)
                v_hat = v_new / (1.0 - beta2 ** step_i)
                values[value_id] = (
                    param - lr * (m_hat / (np.sqrt(v_hat) + eps)) - lr * weight_decay * param
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAM_UPDATE_M":
                arg = step["arg"]
                param = values[input_ids[0]]
                grad = values[input_ids[1]]
                m = values[input_ids[2]]
                beta1 = float(arg["beta1"])
                weight_decay = float(arg.get("weight_decay", 0.0))
                grad_eff = grad + weight_decay * param
                values[value_id] = (
                    beta1 * m + (1.0 - beta1) * grad_eff
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAM_UPDATE_V":
                arg = step["arg"]
                param = values[input_ids[0]]
                grad = values[input_ids[1]]
                v = values[input_ids[2]]
                beta2 = float(arg["beta2"])
                weight_decay = float(arg.get("weight_decay", 0.0))
                grad_eff = grad + weight_decay * param
                values[value_id] = (
                    beta2 * v + (1.0 - beta2) * (grad_eff * grad_eff)
                ).astype(np.dtype(step["dtype"]), copy=False)
            elif op == "ADAM_UPDATE_PARAM":
                arg = step["arg"]
                param = values[input_ids[0]]
                m_new = values[input_ids[1]]
                v_new = values[input_ids[2]]
                lr = float(arg["lr"])
                beta1 = float(arg["beta1"])
                beta2 = float(arg["beta2"])
                eps = float(arg["eps"])
                step_i = int(arg["step"])
                m_hat = m_new / (1.0 - beta1 ** step_i)
                v_hat = v_new / (1.0 - beta2 ** step_i)
                values[value_id] = (
                    param - lr * (m_hat / (np.sqrt(v_hat) + eps))
                ).astype(np.dtype(step["dtype"]), copy=False)
            else:
                raise ValueError(f"mock tensor plan: unsupported op {op}")
        root_id = plan.get("root_id", plan.get("rootId"))
        self.calls.append(("run_tensor_plan", root_id))
        return values[root_id].astype(np.dtype(dtype), copy=False).tobytes()

    def run_tensor_plan_resident(self, plan, inputs, dtype):
        self.tensor_plan_resident_count += 1
        data = self.run_tensor_plan(plan, inputs, dtype)
        arr = np.frombuffer(data, dtype=np.dtype(dtype)).copy()
        root_id = plan.get("root_id", plan.get("rootId"))
        hh = self._mint(arr)
        self.calls.append(("run_tensor_plan_resident", root_id, hh))
        return hh
`;

describe("PRD-011.5 WebGPU realizer seam", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Re-register a fresh mock bridge per test so counters start at zero.
    await target.run(`
import browsergrad_jit as bg
${MOCK_BRIDGE_PY}
_mock = MockBridge()
bg.register_webgpu_bridge(_mock)
`);
  });

  it("realizes a single matmul through the bridge and matches NumPy", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      upload: number;
      matmul: number;
      materialize: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
rng = np.random.RandomState(0)
A = rng.uniform(-1, 1, size=(3, 4)).astype(np.float32)
B = rng.uniform(-1, 1, size=(4, 5)).astype(np.float32)
a = bg.from_numpy(A.copy())
b = bg.from_numpy(B.copy())
y_ref = (a @ b).numpy()        # NumPy realizer
y_gpu = bg.realize_webgpu(a @ b)
{
    "max_diff": float(np.max(np.abs(y_ref - y_gpu))),
    "upload": _mock.upload_count,
    "matmul": _mock.matmul_count,
    "materialize": _mock.materialize_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.upload).toBe(2); // A, B
    expect(result.matmul).toBe(1);
    expect(result.materialize).toBe(1);
  });

  it("residency contract: chained matmuls upload 3 inputs and materialise 1 output", async () => {
    // (X @ W1) @ W2 — the (X @ W1) intermediate must STAY on the GPU and
    // get released immediately after the second matmul consumes it. The
    // mock bridge's alive-set + release counter prove this.
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      upload: number;
      matmul: number;
      materialize: number;
      release: number;
      alive_after: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
rng = np.random.RandomState(1)
X = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
W1 = rng.uniform(-1, 1, size=(3, 4)).astype(np.float32)
W2 = rng.uniform(-1, 1, size=(4, 5)).astype(np.float32)

x = bg.from_numpy(X.copy())
w1 = bg.from_numpy(W1.copy())
w2 = bg.from_numpy(W2.copy())

# Reference via NumPy realizer.
y_ref = ((x @ w1) @ w2).numpy()

# GPU realizer with mock bridge.
y_gpu = bg.realize_webgpu((x @ w1) @ w2)

# Three seed buffers uploaded once each.
# Two matmul calls.
# One materialise.
# Intermediate matmul output should be released — total releases:
#   1 (intermediate) + 1 (root materialise post-call) = 2 in v0.
gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
alive_after = gbt.stats()["handles_alive"]

{
    "max_diff": float(np.max(np.abs(y_ref - y_gpu))),
    "upload": _mock.upload_count,
    "matmul": _mock.matmul_count,
    "materialize": _mock.materialize_count,
    "release": _mock.release_count,
    "alive_after": alive_after,
}
`);
    expect(result.max_diff).toBeLessThan(1e-5);
    expect(result.upload).toBe(3);
    expect(result.matmul).toBe(2);
    expect(result.materialize).toBe(1);
    // Two intermediates: the (X @ W1) handle plus the root after materialise.
    expect(result.release).toBe(2);
    // After the realize call, the only handles still alive are the three
    // seed BUFFERs (X, W1, W2) which persist for cross-call caching.
    expect(result.alive_after).toBe(3);
  });

  it("realize_tensor_plan_webgpu uses one generic plan call, not per-op bridge calls", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      tensor_plan: number;
      upload: number;
      matmul: number;
      fused: number;
      materialize: number;
      root_id: number;
      step_ids: number[];
      ops: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
rng = np.random.RandomState(11)
A = rng.uniform(-0.5, 0.5, size=(3, 4)).astype(np.float32)
B = rng.uniform(-0.5, 0.5, size=(4, 5)).astype(np.float32)
a = bg.from_numpy(A.copy())
b = bg.from_numpy(B.copy())
y = a @ b
out = bg.exp(y) * y
ref = out.numpy()
gpu = bg.realize_tensor_plan_webgpu(out)
plan = bg.gpu_plan_summary(out)
{
    "max_diff": float(np.max(np.abs(ref - gpu))),
    "tensor_plan": _mock.tensor_plan_count,
    "upload": _mock.upload_count,
    "matmul": _mock.matmul_count,
    "fused": _mock.fused_count,
    "materialize": _mock.materialize_count,
    "root_id": int(plan["root_id"]),
    "step_ids": [int(step["value_id"]) for step in plan["steps"]],
    "ops": plan["ops"],
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(1);
    expect(result.upload).toBe(0);
    expect(result.matmul).toBe(0);
    expect(result.fused).toBe(0);
    expect(result.materialize).toBe(0);
    expect(result.ops).toContain("FUSED_ELEMENTWISE");
    expect(result.ops).not.toContain("EXP");
    expect(result.root_id).toBe(result.step_ids[result.step_ids.length - 1]);
    expect(result.step_ids).toEqual([...result.step_ids.keys()]);
  });

  it("realize_tensor_plan_webgpu_resident keeps roots on GPU until explicit numpy boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      before_numpy_materialize: number;
      after_numpy_materialize: number;
      tensor_plan: number;
      tensor_plan_resident: number;
      first_materialized: boolean;
      second_materialized_before_numpy: boolean;
      first_gpu_registered: boolean;
      second_gpu_registered: boolean;
      arr: number[][];
    }>(`
import browsergrad_jit as bg

a = bg.tensor([[1.0, 2.0], [3.0, 4.0]])
b = bg.tensor([[2.0, 0.0], [1.0, 2.0]])
offset = bg.tensor([[3.0, 3.0], [3.0, 3.0]])
first = bg.realize_tensor_plan_webgpu_resident(a @ b)
first_id = first._uop.inputs[0].arg
second = bg.realize_tensor_plan_webgpu_resident(first + offset)
second_id = second._uop.inputs[0].arg
bt = second._get_session().buffer_table
gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
before_numpy = _mock.materialize_count
arr = second.numpy()
{
    "before_numpy_materialize": before_numpy,
    "after_numpy_materialize": _mock.materialize_count,
    "tensor_plan": _mock.tensor_plan_count,
    "tensor_plan_resident": _mock.tensor_plan_resident_count,
    "first_materialized": bt.is_materialized(first_id),
    "second_materialized_before_numpy": before_numpy > 0,
    "first_gpu_registered": gbt.has(first_id),
    "second_gpu_registered": gbt.has(second_id),
    "arr": arr.tolist(),
}
`);
    expect(result.before_numpy_materialize).toBe(0);
    expect(result.after_numpy_materialize).toBe(1);
    expect(result.tensor_plan).toBe(2);
    expect(result.tensor_plan_resident).toBe(2);
    expect(result.first_materialized).toBe(false);
    expect(result.second_materialized_before_numpy).toBe(false);
    expect(result.first_gpu_registered).toBe(true);
    expect(result.second_gpu_registered).toBe(true);
    expect(result.arr).toEqual([
      [7, 7],
      [13, 11],
    ]);
  });

  it("realize_tensor_plan_webgpu handles reshape, permute, reduce, and broadcast plan ops", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      reduce_diff: number;
      broadcast_diff: number;
      tensor_plan: number;
      ops: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_BROADCAST_TO
from browsergrad_jit._tensor_proxy import TensorProxy

rng = np.random.RandomState(12)
x_np = rng.uniform(-1, 1, size=(6,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
reduced = x.reshape(2, 3).permute(1, 0).sum(axis=1, keepdims=True)
reduced_gpu = bg.realize_tensor_plan_webgpu(reduced)
reduced_ref = reduced.numpy()

base_np = np.array([[1.0], [2.0]], dtype=np.float32)
base = bg.from_numpy(base_np.copy())
b_uop = UOp(
    op=OP_BROADCAST_TO,
    inputs=(base._uop,),
    shape=(2, 3),
    dtype="float32",
    arg={"shape": (2, 3)},
)
broadcasted = TensorProxy(b_uop, session=base._get_session(), requires_grad=False)
broadcast_gpu = bg.realize_tensor_plan_webgpu(broadcasted)
broadcast_ref = broadcasted.numpy()
plan = bg.gpu_plan_summary(reduced)
{
    "reduce_diff": float(np.max(np.abs(reduced_gpu - reduced_ref))),
    "broadcast_diff": float(np.max(np.abs(broadcast_gpu - broadcast_ref))),
    "tensor_plan": _mock.tensor_plan_count,
    "ops": plan["ops"],
}
`);
    expect(result.reduce_diff).toBeLessThan(1e-6);
    expect(result.broadcast_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(2);
    expect(result.ops).toContain("RESHAPE");
    expect(result.ops).toContain("PERMUTE");
    expect(result.ops).toContain("REDUCE");
  });

  it("backward(device='webgpu') realizes symbolic leaf grads through tensor-plan WebGPU", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      x_diff: number;
      w_diff: number;
      tensor_plan: number;
      upload: number;
      materialize: number;
      legacy_matmul: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

rng = np.random.RandomState(18)
x_np = rng.uniform(-1, 1, size=(3, 4)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(4, 5)).astype(np.float32)

x_cpu = bg.from_numpy(x_np.copy(), requires_grad=True)
w_cpu = bg.from_numpy(w_np.copy(), requires_grad=True)
((x_cpu @ w_cpu).sum()).backward()

x_gpu = bg.from_numpy(x_np.copy(), requires_grad=True)
w_gpu = bg.from_numpy(w_np.copy(), requires_grad=True)
((x_gpu @ w_gpu).sum()).backward(device="webgpu")

{
    "x_diff": float(np.max(np.abs(x_gpu.grad.numpy() - x_cpu.grad.numpy()))),
    "w_diff": float(np.max(np.abs(w_gpu.grad.numpy() - w_cpu.grad.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "upload": _mock.upload_count,
    "materialize": _mock.materialize_count,
    "legacy_matmul": _mock.matmul_count,
}
`);
    expect(result.x_diff).toBeLessThan(1e-5);
    expect(result.w_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(2);
    expect(result.upload).toBe(0);
    expect(result.materialize).toBe(0);
    expect(result.legacy_matmul).toBe(0);
  });

  it("backward(device='webgpu') refuses closure-only ops instead of falling back to CPU", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ message: string; tensor_plan: number }>(`
import browsergrad_jit as bg

x = bg.tensor([-1.0, 2.0, -3.0], requires_grad=True)
try:
    x.abs().sum().backward(device="webgpu")
    msg = "no_error"
except Exception as e:
    msg = str(e)
{"message": msg, "tensor_plan": _mock.tensor_plan_count}
`);
    expect(result.message).toMatch(/requires symbolic VJP coverage/);
    expect(result.message).toMatch(/CUSTOM/);
    expect(result.tensor_plan).toBe(0);
  });

  it("realize_tensor_plan_webgpu runs functional SGD_UPDATE through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      tensor_plan: number;
      ops: string[];
      upload: number;
      materialize: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

rng = np.random.RandomState(17)
p_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
g_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
p = bg.from_numpy(p_np.copy())
g = bg.from_numpy(g_np.copy())
updated = bg.optim.sgd_update(p, g, lr=0.125, weight_decay=0.01)
gpu = bg.realize_tensor_plan_webgpu(updated)
ref = updated.numpy()
plan = bg.gpu_plan_summary(updated)
{
    "max_diff": float(np.max(np.abs(gpu - ref))),
    "tensor_plan": _mock.tensor_plan_count,
    "ops": plan["ops"],
    "upload": _mock.upload_count,
    "materialize": _mock.materialize_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(1);
    expect(result.ops).toContain("SGD_UPDATE");
    expect(result.upload).toBe(0);
    expect(result.materialize).toBe(0);
  });

  it("realize_tensor_plan_webgpu runs functional AdamW update through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      p_diff: number;
      m_diff: number;
      v_diff: number;
      tensor_plan: number;
      p_ops: string[];
      m_ops: string[];
      v_ops: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np

rng = np.random.RandomState(18)
p_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
g_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
m_np = rng.uniform(-0.1, 0.1, size=(2, 3)).astype(np.float32)
v_np = rng.uniform(0.0, 0.1, size=(2, 3)).astype(np.float32)
p = bg.from_numpy(p_np.copy())
g = bg.from_numpy(g_np.copy())
m = bg.from_numpy(m_np.copy())
v = bg.from_numpy(v_np.copy())
new_p, new_m, new_v = bg.optim.adamw_update(
    p, g, m, v,
    lr=0.01,
    betas=(0.8, 0.95),
    eps=1e-6,
    weight_decay=0.02,
    step=3,
)
p_gpu = bg.realize_tensor_plan_webgpu(new_p)
m_gpu = bg.realize_tensor_plan_webgpu(new_m)
v_gpu = bg.realize_tensor_plan_webgpu(new_v)
{
    "p_diff": float(np.max(np.abs(p_gpu - new_p.numpy()))),
    "m_diff": float(np.max(np.abs(m_gpu - new_m.numpy()))),
    "v_diff": float(np.max(np.abs(v_gpu - new_v.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "p_ops": bg.gpu_plan_summary(new_p)["ops"],
    "m_ops": bg.gpu_plan_summary(new_m)["ops"],
    "v_ops": bg.gpu_plan_summary(new_v)["ops"],
}
`);
    expect(result.p_diff).toBeLessThan(1e-6);
    expect(result.m_diff).toBeLessThan(1e-6);
    expect(result.v_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(3);
    expect(result.p_ops).toContain("ADAMW_UPDATE_PARAM");
    expect(result.m_ops).toContain("ADAMW_UPDATE_M");
    expect(result.v_ops).toContain("ADAMW_UPDATE_V");
  });

  it("realize_tensor_plan_webgpu runs functional Adam update through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      p_diff: number;
      m_diff: number;
      v_diff: number;
      tensor_plan: number;
      p_ops: string[];
      m_ops: string[];
      v_ops: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np

rng = np.random.RandomState(23)
p_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
g_np = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
m_np = rng.uniform(-0.1, 0.1, size=(2, 3)).astype(np.float32)
v_np = rng.uniform(0.0, 0.1, size=(2, 3)).astype(np.float32)
p = bg.from_numpy(p_np.copy())
g = bg.from_numpy(g_np.copy())
m = bg.from_numpy(m_np.copy())
v = bg.from_numpy(v_np.copy())
new_p, new_m, new_v = bg.optim.adam_update(
    p, g, m, v,
    lr=0.02,
    betas=(0.75, 0.9),
    eps=1e-6,
    weight_decay=0.03,
    step=4,
)
p_gpu = bg.realize_tensor_plan_webgpu(new_p)
m_gpu = bg.realize_tensor_plan_webgpu(new_m)
v_gpu = bg.realize_tensor_plan_webgpu(new_v)
{
    "p_diff": float(np.max(np.abs(p_gpu - new_p.numpy()))),
    "m_diff": float(np.max(np.abs(m_gpu - new_m.numpy()))),
    "v_diff": float(np.max(np.abs(v_gpu - new_v.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "p_ops": bg.gpu_plan_summary(new_p)["ops"],
    "m_ops": bg.gpu_plan_summary(new_m)["ops"],
    "v_ops": bg.gpu_plan_summary(new_v)["ops"],
}
`);
    expect(result.p_diff).toBeLessThan(1e-6);
    expect(result.m_diff).toBeLessThan(1e-6);
    expect(result.v_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(3);
    expect(result.p_ops).toContain("ADAM_UPDATE_PARAM");
    expect(result.m_ops).toContain("ADAM_UPDATE_M");
    expect(result.v_ops).toContain("ADAM_UPDATE_V");
  });

  it("Optimizer.step(device='webgpu') routes SGD/Adam/AdamW updates through tensor-plan WebGPU", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      sgd_diff: number;
      adam_p_diff: number;
      adam_m_diff: number;
      adam_v_diff: number;
      adamw_p_diff: number;
      adamw_m_diff: number;
      adamw_v_diff: number;
      tensor_plan: number;
      upload: number;
      materialize: number;
      legacy_matmul: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

p_np = np.array([[0.5, -1.0, 2.0], [1.5, -0.25, 0.75]], dtype=np.float32)
g1_np = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float32)
g2_np = np.array([[-0.3, 0.2, -0.1], [0.05, -0.15, 0.25]], dtype=np.float32)

def run_sgd(device=None):
    p = bg.from_numpy(p_np.copy(), requires_grad=True)
    p.grad = bg.from_numpy(g1_np.copy())
    opt = bg.optim.SGD([p], lr=0.125, weight_decay=0.01)
    if device is None:
        opt.step()
    else:
        opt.step(device=device)
    return p.numpy()

def run_adam(cls, device=None):
    p = bg.from_numpy(p_np.copy(), requires_grad=True)
    opt = cls([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
    for grad_np in (g1_np, g2_np):
        p.grad = bg.from_numpy(grad_np.copy())
        if device is None:
            opt.step()
        else:
            opt.step(device=device)
    return p.numpy(), opt._m[id(p)], opt._v[id(p)]

sgd_cpu = run_sgd()
sgd_gpu = run_sgd("webgpu")

adam_cpu_p, adam_cpu_m, adam_cpu_v = run_adam(bg.optim.Adam)
adam_gpu_p, adam_gpu_m, adam_gpu_v = run_adam(bg.optim.Adam, "webgpu")

adamw_cpu_p, adamw_cpu_m, adamw_cpu_v = run_adam(bg.optim.AdamW)
adamw_gpu_p, adamw_gpu_m, adamw_gpu_v = run_adam(bg.optim.AdamW, "webgpu")

{
    "sgd_diff": float(np.max(np.abs(sgd_gpu - sgd_cpu))),
    "adam_p_diff": float(np.max(np.abs(adam_gpu_p - adam_cpu_p))),
    "adam_m_diff": float(np.max(np.abs(adam_gpu_m - adam_cpu_m))),
    "adam_v_diff": float(np.max(np.abs(adam_gpu_v - adam_cpu_v))),
    "adamw_p_diff": float(np.max(np.abs(adamw_gpu_p - adamw_cpu_p))),
    "adamw_m_diff": float(np.max(np.abs(adamw_gpu_m - adamw_cpu_m))),
    "adamw_v_diff": float(np.max(np.abs(adamw_gpu_v - adamw_cpu_v))),
    "tensor_plan": _mock.tensor_plan_count,
    "upload": _mock.upload_count,
    "materialize": _mock.materialize_count,
    "legacy_matmul": _mock.matmul_count,
}
`);
    expect(result.sgd_diff).toBeLessThan(1e-6);
    expect(result.adam_p_diff).toBeLessThan(1e-5);
    expect(result.adam_m_diff).toBeLessThan(1e-6);
    expect(result.adam_v_diff).toBeLessThan(1e-6);
    expect(result.adamw_p_diff).toBeLessThan(1e-5);
    expect(result.adamw_m_diff).toBeLessThan(1e-6);
    expect(result.adamw_v_diff).toBeLessThan(1e-6);
    expect(result.tensor_plan).toBe(13);
    expect(result.upload).toBe(0);
    expect(result.materialize).toBe(0);
    expect(result.legacy_matmul).toBe(0);
  });

  it("SGD.step(device='webgpu') refuses momentum until momentum state is GPU IR", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ message: string; tensor_plan: number }>(`
import browsergrad_jit as bg
import numpy as np

p = bg.from_numpy(np.array([1.0, -2.0], dtype=np.float32), requires_grad=True)
p.grad = bg.from_numpy(np.array([0.25, -0.5], dtype=np.float32))
opt = bg.optim.SGD([p], lr=0.1, momentum=0.9)
try:
    opt.step(device="webgpu")
    msg = "no_error"
except Exception as e:
    msg = str(e)
{"message": msg, "tensor_plan": _mock.tensor_plan_count}
`);
    expect(result.message).toMatch(/does not support momentum/);
    expect(result.tensor_plan).toBe(0);
  });

  it("backward(device='webgpu', resident=True) and SGD.step(..., resident=True) avoid CPU readback", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      before_materialize: number;
      after_materialize: number;
      grad_materialized_before_step: boolean;
      param_materialized_after_step: boolean;
      grad_materialized_after_param_numpy: boolean;
      grad_gpu_registered: boolean;
      param_gpu_registered: boolean;
      tensor_plan: number;
      tensor_plan_resident: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

p_np = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)

p_cpu = bg.from_numpy(p_np.copy(), requires_grad=True)
loss_cpu = (p_cpu * p_cpu).sum()
loss_cpu.backward()
bg.optim.SGD([p_cpu], lr=0.25).step()
expected = p_cpu.numpy()

p_gpu = bg.from_numpy(p_np.copy(), requires_grad=True)
loss_gpu = (p_gpu * p_gpu).sum()
loss_gpu.backward(device="webgpu", resident=True)
grad_id = p_gpu.grad._uop.inputs[0].arg
bt = p_gpu._get_session().buffer_table
gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
grad_materialized_before_step = bt.is_materialized(grad_id)

opt = bg.optim.SGD([p_gpu], lr=0.25)
before_materialize = _mock.materialize_count
opt.step(device="webgpu", resident=True)
param_id = p_gpu._uop.inputs[0].arg
param_materialized_after_step = bt.is_materialized(param_id)
actual = p_gpu.numpy()

{
    "max_diff": float(np.max(np.abs(actual - expected))),
    "before_materialize": before_materialize,
    "after_materialize": _mock.materialize_count,
    "grad_materialized_before_step": grad_materialized_before_step,
    "param_materialized_after_step": param_materialized_after_step,
    "grad_materialized_after_param_numpy": bt.is_materialized(grad_id),
    "grad_gpu_registered": gbt.has(grad_id),
    "param_gpu_registered": gbt.has(param_id),
    "tensor_plan": _mock.tensor_plan_count,
    "tensor_plan_resident": _mock.tensor_plan_resident_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.before_materialize).toBe(0);
    expect(result.after_materialize).toBe(1);
    expect(result.grad_materialized_before_step).toBe(false);
    expect(result.param_materialized_after_step).toBe(false);
    expect(result.grad_materialized_after_param_numpy).toBe(false);
    expect(result.grad_gpu_registered).toBe(true);
    expect(result.param_gpu_registered).toBe(true);
    expect(result.tensor_plan).toBe(2);
    expect(result.tensor_plan_resident).toBe(2);
  });

  it("Adam/AdamW step(device='webgpu', resident=True) keep optimizer state resident", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      adam_p_diff: number;
      adam_m_diff: number;
      adam_v_diff: number;
      adamw_p_diff: number;
      adamw_m_diff: number;
      adamw_v_diff: number;
      before_materialize: number;
      after_materialize: number;
      adam_param_materialized_before_numpy: boolean;
      adam_m_materialized_before_numpy: boolean;
      adam_v_materialized_before_numpy: boolean;
      adamw_param_materialized_before_numpy: boolean;
      adamw_m_materialized_before_numpy: boolean;
      adamw_v_materialized_before_numpy: boolean;
      adam_param_gpu_registered: boolean;
      adam_m_gpu_registered: boolean;
      adam_v_gpu_registered: boolean;
      adamw_param_gpu_registered: boolean;
      adamw_m_gpu_registered: boolean;
      adamw_v_gpu_registered: boolean;
      tensor_plan: number;
      tensor_plan_resident: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

p_np = np.array([[0.5, -1.0, 2.0], [1.5, -0.25, 0.75]], dtype=np.float32)
g1_np = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float32)
g2_np = np.array([[-0.3, 0.2, -0.1], [0.05, -0.15, 0.25]], dtype=np.float32)

def run_cpu(cls):
    p = bg.from_numpy(p_np.copy(), requires_grad=True)
    opt = cls([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
    for grad_np in (g1_np, g2_np):
        p.grad = bg.from_numpy(grad_np.copy())
        opt.step()
    return p.numpy(), opt._m[id(p)], opt._v[id(p)]

def run_resident(cls):
    p = bg.from_numpy(p_np.copy(), requires_grad=True)
    opt = cls([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
    for grad_np in (g1_np, g2_np):
        p.grad = bg.realize_tensor_plan_webgpu_resident(bg.from_numpy(grad_np.copy()))
        opt.step(device="webgpu", resident=True)
    pid = id(p)
    bt = p._get_session().buffer_table
    gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
    param_id = p._uop.inputs[0].arg
    m_id = opt._m_resident[pid]._uop.inputs[0].arg
    v_id = opt._v_resident[pid]._uop.inputs[0].arg
    before = {
        "param_materialized": bt.is_materialized(param_id),
        "m_materialized": bt.is_materialized(m_id),
        "v_materialized": bt.is_materialized(v_id),
        "param_registered": gbt.has(param_id),
        "m_registered": gbt.has(m_id),
        "v_registered": gbt.has(v_id),
    }
    return p, opt, before

adam_cpu_p, adam_cpu_m, adam_cpu_v = run_cpu(bg.optim.Adam)
adam_p, adam_opt, adam_before = run_resident(bg.optim.Adam)
adamw_cpu_p, adamw_cpu_m, adamw_cpu_v = run_cpu(bg.optim.AdamW)
adamw_p, adamw_opt, adamw_before = run_resident(bg.optim.AdamW)

before_materialize = _mock.materialize_count
adam_p_np = adam_p.numpy()
adam_m_np = adam_opt._m_resident[id(adam_p)].numpy()
adam_v_np = adam_opt._v_resident[id(adam_p)].numpy()
adamw_p_np = adamw_p.numpy()
adamw_m_np = adamw_opt._m_resident[id(adamw_p)].numpy()
adamw_v_np = adamw_opt._v_resident[id(adamw_p)].numpy()

{
    "adam_p_diff": float(np.max(np.abs(adam_p_np - adam_cpu_p))),
    "adam_m_diff": float(np.max(np.abs(adam_m_np - adam_cpu_m))),
    "adam_v_diff": float(np.max(np.abs(adam_v_np - adam_cpu_v))),
    "adamw_p_diff": float(np.max(np.abs(adamw_p_np - adamw_cpu_p))),
    "adamw_m_diff": float(np.max(np.abs(adamw_m_np - adamw_cpu_m))),
    "adamw_v_diff": float(np.max(np.abs(adamw_v_np - adamw_cpu_v))),
    "before_materialize": before_materialize,
    "after_materialize": _mock.materialize_count,
    "adam_param_materialized_before_numpy": adam_before["param_materialized"],
    "adam_m_materialized_before_numpy": adam_before["m_materialized"],
    "adam_v_materialized_before_numpy": adam_before["v_materialized"],
    "adamw_param_materialized_before_numpy": adamw_before["param_materialized"],
    "adamw_m_materialized_before_numpy": adamw_before["m_materialized"],
    "adamw_v_materialized_before_numpy": adamw_before["v_materialized"],
    "adam_param_gpu_registered": adam_before["param_registered"],
    "adam_m_gpu_registered": adam_before["m_registered"],
    "adam_v_gpu_registered": adam_before["v_registered"],
    "adamw_param_gpu_registered": adamw_before["param_registered"],
    "adamw_m_gpu_registered": adamw_before["m_registered"],
    "adamw_v_gpu_registered": adamw_before["v_registered"],
    "tensor_plan": _mock.tensor_plan_count,
    "tensor_plan_resident": _mock.tensor_plan_resident_count,
}
`);
    expect(result.adam_p_diff).toBeLessThan(1e-5);
    expect(result.adam_m_diff).toBeLessThan(1e-6);
    expect(result.adam_v_diff).toBeLessThan(1e-6);
    expect(result.adamw_p_diff).toBeLessThan(1e-5);
    expect(result.adamw_m_diff).toBeLessThan(1e-6);
    expect(result.adamw_v_diff).toBeLessThan(1e-6);
    expect(result.before_materialize).toBe(0);
    expect(result.after_materialize).toBe(6);
    expect(result.adam_param_materialized_before_numpy).toBe(false);
    expect(result.adam_m_materialized_before_numpy).toBe(false);
    expect(result.adam_v_materialized_before_numpy).toBe(false);
    expect(result.adamw_param_materialized_before_numpy).toBe(false);
    expect(result.adamw_m_materialized_before_numpy).toBe(false);
    expect(result.adamw_v_materialized_before_numpy).toBe(false);
    expect(result.adam_param_gpu_registered).toBe(true);
    expect(result.adam_m_gpu_registered).toBe(true);
    expect(result.adam_v_gpu_registered).toBe(true);
    expect(result.adamw_param_gpu_registered).toBe(true);
    expect(result.adamw_m_gpu_registered).toBe(true);
    expect(result.adamw_v_gpu_registered).toBe(true);
    expect(result.tensor_plan).toBe(16);
    expect(result.tensor_plan_resident).toBe(16);
  });

  it("GPU-resident tensors make default backward and optimizer step stay resident", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      param_diff: number;
      m_diff: number;
      v_diff: number;
      before_materialize: number;
      after_materialize: number;
      grad_materialized: boolean;
      param_materialized: boolean;
      m_materialized: boolean;
      v_materialized: boolean;
      grad_gpu_registered: boolean;
      param_gpu_registered: boolean;
      m_gpu_registered: boolean;
      v_gpu_registered: boolean;
      tensor_plan: number;
      tensor_plan_resident: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

p_np = np.array([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32)
zero = np.zeros_like(p_np)

p_cpu = bg.from_numpy(p_np.copy(), requires_grad=True)
loss_cpu = (p_cpu * p_cpu).sum()
loss_cpu.backward()
opt_cpu = bg.optim.AdamW([p_cpu], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
opt_cpu.step()
expected_p = p_cpu.numpy()
expected_m = opt_cpu._m[id(p_cpu)]
expected_v = opt_cpu._v[id(p_cpu)]

p_seed = bg.from_numpy(p_np.copy(), requires_grad=True)
p = bg.realize_tensor_plan_webgpu_resident(p_seed + bg.from_numpy(zero.copy()))
loss = (p * p).sum()
loss.backward()
opt = bg.optim.AdamW([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
opt.step()

pid = id(p)
bt = p._get_session().buffer_table
gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
grad_id = p.grad._uop.inputs[0].arg
param_id = p._uop.inputs[0].arg
m_id = opt._m_resident[pid]._uop.inputs[0].arg
v_id = opt._v_resident[pid]._uop.inputs[0].arg
before_materialize = _mock.materialize_count
status = {
    "grad_materialized": bt.is_materialized(grad_id),
    "param_materialized": bt.is_materialized(param_id),
    "m_materialized": bt.is_materialized(m_id),
    "v_materialized": bt.is_materialized(v_id),
    "grad_registered": gbt.has(grad_id),
    "param_registered": gbt.has(param_id),
    "m_registered": gbt.has(m_id),
    "v_registered": gbt.has(v_id),
}
actual_p = p.numpy()
actual_m = opt._m_resident[pid].numpy()
actual_v = opt._v_resident[pid].numpy()
{
    "param_diff": float(np.max(np.abs(actual_p - expected_p))),
    "m_diff": float(np.max(np.abs(actual_m - expected_m))),
    "v_diff": float(np.max(np.abs(actual_v - expected_v))),
    "before_materialize": before_materialize,
    "after_materialize": _mock.materialize_count,
    "grad_materialized": status["grad_materialized"],
    "param_materialized": status["param_materialized"],
    "m_materialized": status["m_materialized"],
    "v_materialized": status["v_materialized"],
    "grad_gpu_registered": status["grad_registered"],
    "param_gpu_registered": status["param_registered"],
    "m_gpu_registered": status["m_registered"],
    "v_gpu_registered": status["v_registered"],
    "tensor_plan": _mock.tensor_plan_count,
    "tensor_plan_resident": _mock.tensor_plan_resident_count,
}
`);
    expect(result.param_diff).toBeLessThan(1e-5);
    expect(result.m_diff).toBeLessThan(1e-6);
    expect(result.v_diff).toBeLessThan(1e-6);
    expect(result.before_materialize).toBe(0);
    expect(result.after_materialize).toBe(3);
    expect(result.grad_materialized).toBe(false);
    expect(result.param_materialized).toBe(false);
    expect(result.m_materialized).toBe(false);
    expect(result.v_materialized).toBe(false);
    expect(result.grad_gpu_registered).toBe(true);
    expect(result.param_gpu_registered).toBe(true);
    expect(result.m_gpu_registered).toBe(true);
    expect(result.v_gpu_registered).toBe(true);
    expect(result.tensor_plan_resident).toBeGreaterThanOrEqual(5);
    expect(result.tensor_plan).toBe(result.tensor_plan_resident);
  });

  it("realize_tensor_plan_webgpu runs LayerNorm forward/backward roots through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      out_diff: number;
      gx_diff: number;
      gw_diff: number;
      gb_diff: number;
      tensor_plan: number;
      out_ops: string[];
      gx_ops: string[];
      gw_ops: string[];
      gb_ops: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(31)
x_np = rng.uniform(-1, 1, size=(2, 3, 4)).astype(np.float32)
w_np = rng.uniform(0.5, 1.5, size=(4,)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(4,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.layer_norm(x, (4,), w, b, eps=1e-5)
out_gpu = bg.realize_tensor_plan_webgpu(out)
dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)
gx_gpu = bg.realize_tensor_plan_webgpu(gx)
gw_gpu = bg.realize_tensor_plan_webgpu(gw)
gb_gpu = bg.realize_tensor_plan_webgpu(gb)
{
    "out_diff": float(np.max(np.abs(out_gpu - out.numpy()))),
    "gx_diff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gw_diff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gb_diff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "out_ops": bg.gpu_plan_summary(out)["ops"],
    "gx_ops": bg.gpu_plan_summary(gx)["ops"],
    "gw_ops": bg.gpu_plan_summary(gw)["ops"],
    "gb_ops": bg.gpu_plan_summary(gb)["ops"],
}
`);
    expect(result.out_diff).toBeLessThan(1e-5);
    expect(result.gx_diff).toBeLessThan(1e-5);
    expect(result.gw_diff).toBeLessThan(1e-5);
    expect(result.gb_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(4);
    expect(result.out_ops).toContain("LAYER_NORM");
    expect(result.gx_ops).toContain("LAYER_NORM_BACKWARD_INPUT");
    expect(result.gw_ops).toContain("LAYER_NORM_BACKWARD_WEIGHT");
    expect(result.gb_ops).toContain("LAYER_NORM_BACKWARD_BIAS");
  });

  it("realize_tensor_plan_webgpu runs Conv1d/Conv2d through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      conv1d_diff: number;
      conv2d_diff: number;
      tensor_plan: number;
      conv1d_bridge: number;
      conv2d_bridge: number;
      ops1: string[];
      ops2: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np

rng = np.random.RandomState(13)
x1_np = rng.uniform(-1, 1, size=(1, 4, 9)).astype(np.float32)
w1_np = rng.uniform(-1, 1, size=(6, 2, 3)).astype(np.float32)
b1_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
x1 = bg.from_numpy(x1_np.copy())
w1 = bg.from_numpy(w1_np.copy())
b1 = bg.from_numpy(b1_np.copy())
out1 = F.conv1d(x1, w1, b1, stride=2, padding=2, dilation=2, groups=2)
gpu1 = bg.realize_tensor_plan_webgpu(out1)
ref1 = out1.numpy()

x2_np = rng.uniform(-1, 1, size=(1, 4, 5, 4)).astype(np.float32)
w2_np = rng.uniform(-1, 1, size=(6, 2, 2, 2)).astype(np.float32)
b2_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
x2 = bg.from_numpy(x2_np.copy())
w2 = bg.from_numpy(w2_np.copy())
b2 = bg.from_numpy(b2_np.copy())
out2 = F.conv2d(x2, w2, b2, stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=2)
gpu2 = bg.realize_tensor_plan_webgpu(out2)
ref2 = out2.numpy()
{
    "conv1d_diff": float(np.max(np.abs(gpu1 - ref1))),
    "conv2d_diff": float(np.max(np.abs(gpu2 - ref2))),
    "tensor_plan": _mock.tensor_plan_count,
    "conv1d_bridge": _mock.conv1d_count,
    "conv2d_bridge": _mock.conv2d_count,
    "ops1": bg.gpu_plan_summary(out1)["ops"],
    "ops2": bg.gpu_plan_summary(out2)["ops"],
}
`);
    expect(result.conv1d_diff).toBeLessThan(1e-5);
    expect(result.conv2d_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(2);
    expect(result.conv1d_bridge).toBe(0);
    expect(result.conv2d_bridge).toBe(0);
    expect(result.ops1).toContain("CONV1D");
    expect(result.ops2).toContain("CONV2D");
  });

  it("realize_tensor_plan_webgpu runs Conv3d forward/backward roots through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      out_diff: number;
      gx_diff: number;
      gw_diff: number;
      gb_diff: number;
      tensor_plan: number;
      out_ops: string[];
      gx_ops: string[];
      gw_ops: string[];
      gb_ops: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(16)
x_np = rng.uniform(-1, 1, size=(1, 4, 4, 5, 4)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 2, 2, 2)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv3d(x, w, b, stride=(1, 1, 1), padding=(1, 1, 0), dilation=(1, 2, 1), groups=2)
out_gpu = bg.realize_tensor_plan_webgpu(out)
dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)
gx_gpu = bg.realize_tensor_plan_webgpu(gx)
gw_gpu = bg.realize_tensor_plan_webgpu(gw)
gb_gpu = bg.realize_tensor_plan_webgpu(gb)
{
    "out_diff": float(np.max(np.abs(out_gpu - out.numpy()))),
    "gx_diff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gw_diff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gb_diff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "out_ops": bg.gpu_plan_summary(out)["ops"],
    "gx_ops": bg.gpu_plan_summary(gx)["ops"],
    "gw_ops": bg.gpu_plan_summary(gw)["ops"],
    "gb_ops": bg.gpu_plan_summary(gb)["ops"],
}
`);
    expect(result.out_diff).toBeLessThan(1e-5);
    expect(result.gx_diff).toBeLessThan(1e-5);
    expect(result.gw_diff).toBeLessThan(1e-5);
    expect(result.gb_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(4);
    expect(result.out_ops).toContain("CONV3D");
    expect(result.gx_ops).toContain("CONV3D_BACKWARD_INPUT");
    expect(result.gw_ops).toContain("CONV3D_BACKWARD_WEIGHT");
    expect(result.gb_ops).toContain("CONV3D_BACKWARD_BIAS");
  });

  it("realize_tensor_plan_webgpu runs Conv1d backward roots through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      gx_diff: number;
      gw_diff: number;
      gb_diff: number;
      tensor_plan: number;
      legacy_counts: number[];
      gx_ops: string[];
      gw_ops: string[];
      gb_ops: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(14)
x_np = rng.uniform(-1, 1, size=(1, 4, 9)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 3)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv1d(x, w, b, stride=2, padding=2, dilation=2, groups=2)
dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)
gx_gpu = bg.realize_tensor_plan_webgpu(gx)
gw_gpu = bg.realize_tensor_plan_webgpu(gw)
gb_gpu = bg.realize_tensor_plan_webgpu(gb)
{
    "gx_diff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gw_diff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gb_diff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "legacy_counts": [
        _mock.conv1d_backward_input_count,
        _mock.conv1d_backward_weight_count,
        _mock.conv1d_backward_bias_count,
    ],
    "gx_ops": bg.gpu_plan_summary(gx)["ops"],
    "gw_ops": bg.gpu_plan_summary(gw)["ops"],
    "gb_ops": bg.gpu_plan_summary(gb)["ops"],
}
`);
    expect(result.gx_diff).toBeLessThan(1e-5);
    expect(result.gw_diff).toBeLessThan(1e-5);
    expect(result.gb_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(3);
    expect(result.legacy_counts).toEqual([0, 0, 0]);
    expect(result.gx_ops).toContain("CONV1D_BACKWARD_INPUT");
    expect(result.gw_ops).toContain("CONV1D_BACKWARD_WEIGHT");
    expect(result.gb_ops).toContain("CONV1D_BACKWARD_BIAS");
  });

  it("realize_tensor_plan_webgpu runs Conv2d backward roots through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      gx_diff: number;
      gw_diff: number;
      gb_diff: number;
      tensor_plan: number;
      legacy_counts: number[];
      gx_ops: string[];
      gw_ops: string[];
      gb_ops: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(15)
x_np = rng.uniform(-1, 1, size=(1, 4, 5, 4)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 2, 2)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv2d(x, w, b, stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=2)
dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)
gx_gpu = bg.realize_tensor_plan_webgpu(gx)
gw_gpu = bg.realize_tensor_plan_webgpu(gw)
gb_gpu = bg.realize_tensor_plan_webgpu(gb)
{
    "gx_diff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gw_diff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gb_diff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "legacy_counts": [
        _mock.conv2d_backward_input_count,
        _mock.conv2d_backward_weight_count,
        _mock.conv2d_backward_bias_count,
    ],
    "gx_ops": bg.gpu_plan_summary(gx)["ops"],
    "gw_ops": bg.gpu_plan_summary(gw)["ops"],
    "gb_ops": bg.gpu_plan_summary(gb)["ops"],
}
`);
    expect(result.gx_diff).toBeLessThan(1e-5);
    expect(result.gw_diff).toBeLessThan(1e-5);
    expect(result.gb_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(3);
    expect(result.legacy_counts).toEqual([0, 0, 0]);
    expect(result.gx_ops).toContain("CONV2D_BACKWARD_INPUT");
    expect(result.gw_ops).toContain("CONV2D_BACKWARD_WEIGHT");
    expect(result.gb_ops).toContain("CONV2D_BACKWARD_BIAS");
  });

  it("realize_tensor_plan_webgpu runs ConvTranspose2d forward/backward roots through generic plan path", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      out_diff: number;
      gx_diff: number;
      gw_diff: number;
      gb_diff: number;
      tensor_plan: number;
      out_ops: string[];
      gx_ops: string[];
      gw_ops: string[];
      gb_ops: string[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(17)
x_np = rng.uniform(-1, 1, size=(1, 4, 2, 3)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(4, 2, 2, 2)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(4,)).astype(np.float32)
x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv_transpose2d(x, w, b, stride=(2, 1), padding=(1, 0), output_padding=(1, 0), dilation=(2, 1), groups=2)
out_gpu = bg.realize_tensor_plan_webgpu(out)
dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)
gx_gpu = bg.realize_tensor_plan_webgpu(gx)
gw_gpu = bg.realize_tensor_plan_webgpu(gw)
gb_gpu = bg.realize_tensor_plan_webgpu(gb)
{
    "out_diff": float(np.max(np.abs(out_gpu - out.numpy()))),
    "gx_diff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gw_diff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gb_diff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "tensor_plan": _mock.tensor_plan_count,
    "out_ops": bg.gpu_plan_summary(out)["ops"],
    "gx_ops": bg.gpu_plan_summary(gx)["ops"],
    "gw_ops": bg.gpu_plan_summary(gw)["ops"],
    "gb_ops": bg.gpu_plan_summary(gb)["ops"],
}
`);
    expect(result.out_diff).toBeLessThan(1e-5);
    expect(result.gx_diff).toBeLessThan(1e-5);
    expect(result.gw_diff).toBeLessThan(1e-5);
    expect(result.gb_diff).toBeLessThan(1e-5);
    expect(result.tensor_plan).toBe(4);
    expect(result.out_ops).toContain("CONV_TRANSPOSE2D");
    expect(result.gx_ops).toContain("CONV_TRANSPOSE2D_BACKWARD_INPUT");
    expect(result.gw_ops).toContain("CONV_TRANSPOSE2D_BACKWARD_WEIGHT");
    expect(result.gb_ops).toContain("CONV_TRANSPOSE2D_BACKWARD_BIAS");
  });

  it("realize_tensor_plan_webgpu refuses CUSTOM ops before bridge dispatch", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      message: string;
      tensor_plan: number;
      flash: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
Q = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
K = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
V = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
out = bg.kernels.flash_attention(Q, K, V)
try:
    bg.realize_tensor_plan_webgpu(out)
    msg = "no_error"
except bg.JitNotImplementedError as e:
    msg = str(e)
{"message": msg, "tensor_plan": _mock.tensor_plan_count, "flash": _mock.flash_count}
`);
    expect(result.message).toMatch(/refuses CUSTOM/);
    expect(result.tensor_plan).toBe(0);
    expect(result.flash).toBe(0);
  });

  it("reuses uploaded seed buffers on a second realize call", async () => {
    // Calling realize_webgpu twice on graphs sharing inputs should not
    // re-upload them — that's the entire point of the GpuBufferTable
    // persisting across calls.
    const target = await getJitTarget();
    const result = await target.run<{
      uploads_first: number;
      uploads_second: number;
      matmul_total: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
A = np.eye(4, dtype=np.float32)
B = np.eye(4, dtype=np.float32) * 2.0
a = bg.from_numpy(A)
b = bg.from_numpy(B)

bg.realize_webgpu(a @ b)
u1 = _mock.upload_count
bg.realize_webgpu(a @ b)
u2 = _mock.upload_count

{"uploads_first": u1, "uploads_second": u2, "matmul_total": _mock.matmul_count}
`);
    expect(result.uploads_first).toBe(2);
    expect(result.uploads_second).toBe(2); // no new uploads on the second call
    expect(result.matmul_total).toBe(2);
  });

  it("realize_webgpu raises if no bridge is registered", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
bg.unregister_webgpu_bridge()
a = bg.from_numpy(np.eye(2, dtype=np.float32))
try:
    bg.realize_webgpu(a @ a)
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/No WebGPU bridge registered/);
    expect(err).toMatch(/register_webgpu_bridge/);
  });

  it("unsupported opcode raises with a pointer to bg.realize()", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
# REDUCE is not in the v0 WebGPU realizer whitelist.
a = bg.from_numpy(np.ones((3, 3), dtype=np.float32))
try:
    bg.realize_webgpu(a.sum())
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/REDUCE/);
    expect(err).toMatch(/bg\.realize\(\)/);
  });

  it("flash_attention CUSTOM op routes through bridge.flash_attention", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      flash_count: number;
      shape: number[];
      max_diff: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

B, H, Sq, Sk, D = 2, 4, 8, 8, 16
scale = 1.0 / np.sqrt(D)
rng = np.random.RandomState(2)
Q_np = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
K_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
V_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)

Q = bg.from_numpy(Q_np.copy())
K = bg.from_numpy(K_np.copy())
V = bg.from_numpy(V_np.copy())

out = bg.kernels.flash_attention(Q, K, V)
arr_gpu = bg.realize_webgpu(out)

# NumPy reference for parity.
scores = np.matmul(Q_np, np.swapaxes(K_np, -1, -2)) * scale
m_ = scores.max(axis=-1, keepdims=True)
e = np.exp(scores - m_)
p = e / e.sum(axis=-1, keepdims=True)
ref = np.matmul(p, V_np)

{
    "flash_count": _mock.flash_count,
    "shape": list(arr_gpu.shape),
    "max_diff": float(np.max(np.abs(arr_gpu - ref))),
}
`);
    expect(result.flash_count).toBe(1);
    expect(result.shape).toEqual([2, 4, 8, 16]);
    expect(result.max_diff).toBeLessThan(1e-4);
  });

  it("flash_attention accepts an additive mask", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      flash_count: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

B, H, Sq, Sk, D = 1, 2, 4, 4, 8
scale = 1.0 / np.sqrt(D)
rng = np.random.RandomState(3)
Q_np = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
K_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
V_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
# Causal mask: lower-triangular zeros, upper-triangular -inf.
mask_np = np.where(
    np.tri(Sq, Sk, dtype=bool),
    0.0,
    -1e9,
).astype(np.float32).reshape(1, 1, Sq, Sk)

Q = bg.from_numpy(Q_np.copy())
K = bg.from_numpy(K_np.copy())
V = bg.from_numpy(V_np.copy())
mask = bg.from_numpy(mask_np.copy())

out = bg.kernels.flash_attention(Q, K, V, mask=mask)
arr_gpu = bg.realize_webgpu(out)

scores = np.matmul(Q_np, np.swapaxes(K_np, -1, -2)) * scale + mask_np
m_ = scores.max(axis=-1, keepdims=True)
e = np.exp(scores - m_)
p = e / e.sum(axis=-1, keepdims=True)
ref = np.matmul(p, V_np)
{
    "max_diff": float(np.max(np.abs(arr_gpu - ref))),
    "flash_count": _mock.flash_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-4);
    expect(result.flash_count).toBe(1);
  });

  it("CONV2D primitive routes through bridge.conv2d and matches NumPy", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      shape: number[];
      op: string;
      conv2d_count: number;
      upload: number;
      materialize: number;
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np

rng = np.random.RandomState(4)
x_np = rng.uniform(-1, 1, size=(1, 4, 5, 4)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 2, 2)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)

x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv2d(x, w, b, stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=2)
gpu = bg.realize_webgpu(out)
ref = out.numpy()

{
    "max_diff": float(np.max(np.abs(gpu - ref))),
    "shape": list(gpu.shape),
    "op": out._uop.op,
    "conv2d_count": _mock.conv2d_count,
    "upload": _mock.upload_count,
    "materialize": _mock.materialize_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-5);
    expect(result.shape).toEqual([1, 6, 7, 3]);
    expect(result.op).toBe("CONV2D");
    expect(result.conv2d_count).toBe(1);
    expect(result.upload).toBe(3);
    expect(result.materialize).toBe(1);
  });

  it("CONV1D primitive and backward routes through WebGPU bridge", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      outDiff: number;
      gxDiff: number;
      gwDiff: number;
      gbDiff: number;
      ops: string[];
      counts: number[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import toposort
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(6)
x_np = rng.uniform(-1, 1, size=(1, 4, 9)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 3)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)

x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
out = F.conv1d(x, w, b, stride=2, padding=2, dilation=2, groups=2)
out_gpu = bg.realize_webgpu(out)
out_ref = out.numpy()

dy_np = rng.uniform(-1, 1, size=out.shape).astype(np.float32)
dy = bg.from_numpy(dy_np.copy())
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)

gx_gpu = bg.realize_webgpu(gx)
gw_gpu = bg.realize_webgpu(gw)
gb_gpu = bg.realize_webgpu(gb)

{
    "outDiff": float(np.max(np.abs(out_gpu - out_ref))),
    "gxDiff": float(np.max(np.abs(gx_gpu - gx.numpy()))),
    "gwDiff": float(np.max(np.abs(gw_gpu - gw.numpy()))),
    "gbDiff": float(np.max(np.abs(gb_gpu - gb.numpy()))),
    "ops": [u.op for u in toposort(out._uop)] + [u.op for u in toposort(gx._uop)] + [u.op for u in toposort(gw._uop)] + [u.op for u in toposort(gb._uop)],
    "counts": [
        _mock.conv1d_count,
        _mock.conv1d_backward_input_count,
        _mock.conv1d_backward_weight_count,
        _mock.conv1d_backward_bias_count,
    ],
}
`);
    expect(result.outDiff).toBeLessThan(1e-5);
    expect(result.gxDiff).toBeLessThan(1e-5);
    expect(result.gwDiff).toBeLessThan(1e-5);
    expect(result.gbDiff).toBeLessThan(1e-5);
    expect(result.ops).toContain("CONV1D");
    expect(result.ops).toContain("CONV1D_BACKWARD_INPUT");
    expect(result.ops).toContain("CONV1D_BACKWARD_WEIGHT");
    expect(result.ops).toContain("CONV1D_BACKWARD_BIAS");
    expect(result.counts).toEqual([1, 1, 1, 1]);
  });

  it("CONV2D backward primitives route through WebGPU bridge and match NumPy", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      gxDiff: number;
      gwDiff: number;
      gbDiff: number;
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
      counts: number[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import toposort
from browsergrad_jit._tensor_proxy import TensorProxy
from browsergrad_jit._vjp import get_rule

rng = np.random.RandomState(5)
x_np = rng.uniform(-1, 1, size=(1, 4, 5, 4)).astype(np.float32)
w_np = rng.uniform(-1, 1, size=(6, 2, 2, 2)).astype(np.float32)
b_np = rng.uniform(-0.2, 0.2, size=(6,)).astype(np.float32)
dy_np = rng.uniform(-1, 1, size=(1, 6, 7, 3)).astype(np.float32)

x = bg.from_numpy(x_np.copy())
w = bg.from_numpy(w_np.copy())
b = bg.from_numpy(b_np.copy())
dy = bg.from_numpy(dy_np.copy())

out = F.conv2d(x, w, b, stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=2)
rule = get_rule(out._uop.op)
gx_uop, gw_uop, gb_uop = rule(out._uop, (x._uop, w._uop, b._uop), dy._uop)
gx = TensorProxy(gx_uop, session=x._get_session(), requires_grad=False)
gw = TensorProxy(gw_uop, session=x._get_session(), requires_grad=False)
gb = TensorProxy(gb_uop, session=x._get_session(), requires_grad=False)

gx_gpu = bg.realize_webgpu(gx)
gw_gpu = bg.realize_webgpu(gw)
gb_gpu = bg.realize_webgpu(gb)
gx_ref = gx.numpy()
gw_ref = gw.numpy()
gb_ref = gb.numpy()

{
    "gxDiff": float(np.max(np.abs(gx_gpu - gx_ref))),
    "gwDiff": float(np.max(np.abs(gw_gpu - gw_ref))),
    "gbDiff": float(np.max(np.abs(gb_gpu - gb_ref))),
    "gxOps": [u.op for u in toposort(gx._uop)],
    "gwOps": [u.op for u in toposort(gw._uop)],
    "gbOps": [u.op for u in toposort(gb._uop)],
    "counts": [
        _mock.conv2d_backward_input_count,
        _mock.conv2d_backward_weight_count,
        _mock.conv2d_backward_bias_count,
    ],
}
`);
    expect(result.gxDiff).toBeLessThan(1e-5);
    expect(result.gwDiff).toBeLessThan(1e-5);
    expect(result.gbDiff).toBeLessThan(1e-5);
    expect(result.gxOps).toContain("CONV2D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV2D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV2D_BACKWARD_BIAS");
    expect(result.counts).toEqual([1, 1, 1]);
  });

  it("supported_opcodes returns the v0 whitelist", async () => {
    const target = await getJitTarget();
    const ops = await target.run<string[]>(`
import browsergrad_jit as bg
sorted(bg.webgpu_supported_opcodes())
`);
    expect(ops).toEqual([
      "BUFFER",
      "CAST",
      "CONST",
      "CONV1D",
      "CONV1D_BACKWARD_BIAS",
      "CONV1D_BACKWARD_INPUT",
      "CONV1D_BACKWARD_WEIGHT",
      "CONV2D",
      "CONV2D_BACKWARD_BIAS",
      "CONV2D_BACKWARD_INPUT",
      "CONV2D_BACKWARD_WEIGHT",
      "CUSTOM",
      "FUSED_ELEMENTWISE",
      "LOAD",
      "MATMUL",
    ]);
  });

  it("is_available is false until a bridge is registered", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ before: boolean; during: boolean; after: boolean }>(`
import browsergrad_jit as bg
${MOCK_BRIDGE_PY}
bg.unregister_webgpu_bridge()
before = bg.webgpu_is_available()
bg.register_webgpu_bridge(MockBridge())
during = bg.webgpu_is_available()
bg.unregister_webgpu_bridge()
after = bg.webgpu_is_available()
{"before": before, "during": during, "after": after}
`);
    expect(result.before).toBe(false);
    expect(result.during).toBe(true);
    expect(result.after).toBe(false);
  });
});
