import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_PAD_CONFORMANCE } from "../../../test-support/framework-pad-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.nn.functional.pad contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("shares exact padding, fill, dtype, ownership, gradient, and refusal semantics", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_PAD_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      op: string;
      arg: { pad_width: number[][]; mode: string; value: number };
      shape: number[];
      values: number[][];
      dtype: string;
      gradient: number[][];
      dtypeCases: Array<{ dtype: string; value: number | boolean }>;
      zeroPad: { shape: number[]; values: number[][]; ownsData: boolean };
      empty: { shape: number[]; values: number[][] };
      integralRequiresGrad: boolean;
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { padding: number; mode: number; value: number };
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32), requires_grad=True)
output = F.pad(
    source,
    tuple(np.int32(value) for value in valid["pad"]),
    value=np.float32(valid["value"]),
)
values = output.numpy().tolist()
output.backward(bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32)))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = bg.from_numpy(np.asarray([[1]], dtype=np.dtype(case["dtype"])))
    padded = F.pad(typed, (1, 0), value=case["value"])
    dtype_cases.append({"dtype": padded.dtype, "value": padded.numpy().tolist()[0][0]})

zero_spec = fixture["zeroPad"]
zero_pad = F.pad(bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32)), tuple())
zero_array = zero_pad.numpy()

empty_spec = fixture["empty"]
empty = F.pad(
    bg.from_numpy(np.empty(tuple(empty_spec["inputShape"]), dtype=np.float32)),
    tuple(empty_spec["pad"]),
    value=empty_spec["value"],
)

integer_source = bg.from_numpy(np.asarray([1, 2], dtype=np.int64), requires_grad=True)
integer_output = F.pad(integer_source, (1, 1))

class HostilePadding:
    calls = 0
    def __len__(self):
        HostilePadding.calls += 1
        return 2
    def __iter__(self):
        HostilePadding.calls += 1
        return iter((1, 1))

class HostileMode:
    calls = 0
    def __eq__(self, other):
        HostileMode.calls += 1
        return True

class HostileValue:
    calls = 0
    def __float__(self):
        HostileValue.calls += 1
        return 0.0
    def __int__(self):
        HostileValue.calls += 1
        return 0

errors = {}
attempts = {
    "non-tensor": lambda: F.pad([[1.0]], (1, 1)),
    "non-sequence": lambda: F.pad(source, (value for value in (1, 1))),
    "odd-length": lambda: F.pad(source, (1,)),
    "too-many-dimensions": lambda: F.pad(source, (1, 1, 1, 1, 1, 1)),
    "bool-padding": lambda: F.pad(source, (True, 1)),
    "float-padding": lambda: F.pad(source, (1.0, 1)),
    "negative-padding": lambda: F.pad(source, (-1, 0)),
    "hostile-padding": lambda: F.pad(source, HostilePadding()),
    "mode-type": lambda: F.pad(source, (1, 1), mode=7),
    "unsupported-mode": lambda: F.pad(source, (1, 1), mode="reflect"),
    "hostile-mode": lambda: F.pad(source, (1, 1), mode=HostileMode()),
    "hostile-value": lambda: F.pad(source, (1, 1), value=HostileValue()),
    "unsupported-dtype": lambda: F.pad(
        bg.from_numpy(np.ones((1,), dtype=np.uint16)), (1, 1)
    ),
    "fractional-integer-fill": lambda: F.pad(
        bg.from_numpy(np.ones((1,), dtype=np.int32)), (1, 1), value=1.5
    ),
    "overflowing-fill": lambda: F.pad(
        bg.from_numpy(np.ones((1,), dtype=np.uint8)), (1, 1), value=256
    ),
    "nonfinite-floating-fill": lambda: F.pad(source, (1, 1), value=float("inf")),
    "zero-size-oversized-axis": lambda: F.pad(
        bg.from_numpy(np.empty((0, 1), dtype=np.float32)),
        (0, (1 << 28)),
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

array = output.numpy()
owns_data = bool(array.flags["OWNDATA"])
array[0, 0] = 999
rerun = F.pad(source, tuple(valid["pad"]), value=valid["value"])

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "arg": dict(output._uop.arg),
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradient": source.grad.numpy().tolist(),
    "dtypeCases": dtype_cases,
    "zeroPad": {
        "shape": list(zero_pad.shape),
        "values": zero_array.tolist(),
        "ownsData": bool(zero_array.flags["OWNDATA"]),
    },
    "empty": {"shape": list(empty.shape), "values": empty.numpy().tolist()},
    "integralRequiresGrad": integer_output.requires_grad,
    "ownsData": owns_data,
    "rerunValues": rerun.numpy().tolist(),
    "errors": errors,
    "hostileCalls": {
        "padding": HostilePadding.calls,
        "mode": HostileMode.calls,
        "value": HostileValue.calls,
    },
}
`);

    expect(result.schema).toBe(FRAMEWORK_PAD_CONFORMANCE.schema);
    expect(result.op).toBe("PAD");
    expect(result.arg).toEqual({
      pad_width: FRAMEWORK_PAD_CONFORMANCE.valid.canonicalPadWidth,
      mode: "constant",
      value: FRAMEWORK_PAD_CONFORMANCE.valid.value,
    });
    expect(result.shape).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradient).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedGradient);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_PAD_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.dtype,
        value: testCase.expectedValue,
      })),
    );
    expect(result.zeroPad).toEqual({
      shape: FRAMEWORK_PAD_CONFORMANCE.zeroPad.expectedShape,
      values: FRAMEWORK_PAD_CONFORMANCE.zeroPad.expectedValues,
      ownsData: true,
    });
    expect(result.empty).toEqual({
      shape: FRAMEWORK_PAD_CONFORMANCE.empty.expectedShape,
      values: FRAMEWORK_PAD_CONFORMANCE.empty.expectedValues,
    });
    expect(result.integralRequiresGrad).toBe(false);
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ padding: 0, mode: 0, value: 0 });
    for (const invalid of FRAMEWORK_PAD_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides typed slice VJP, batch-safe vmap, exact ONNX Pad, and device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_PAD_CONFORMANCE);
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      mappedArg: { pad_width: number[][]; mode: string; value: number };
      onnx: {
        opTypes: string[];
        inputCount: number;
        pads: number[];
        value: number;
        outputDtype: number;
      };
      boolOnnxError: string;
      planError: string;
      webgpuSupported: boolean[];
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
import struct
from browsergrad_jit._ir import OP_PAD, OP_SLICE, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

${ONNX_PROTOBUF_TEST_HELPERS}

def parse_pad(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    op_types = []
    input_count = None
    output_dtype = None
    pads = None
    value = None
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_type = next(
                item.decode("utf-8") for field, kind, item in node_fields
                if field == 4 and kind == 2
            )
            op_types.append(op_type)
            if op_type == "Pad":
                input_count = sum(1 for field, kind, _ in node_fields if field == 1 and kind == 2)
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                item.decode("utf-8") for field, kind, item in tensor_fields
                if field == 8 and kind == 2
            )
            raw = next(item for field, kind, item in tensor_fields if field == 9 and kind == 2)
            if name.startswith("const_pad_width_"):
                pads = [
                    int.from_bytes(raw[index:index + 8], "little", signed=True)
                    for index in range(0, len(raw), 8)
                ]
            elif name.startswith("const_pad_value_"):
                value = struct.unpack("<f", raw)[0]
        elif number == 12 and wire == 2:
            output_dtype = value_info_dtype(payload)
    if input_count is None or pads is None or value is None or output_dtype is None:
        raise RuntimeError("missing exact Pad ONNX structure")
    return {
        "opTypes": op_types,
        "inputCount": input_count,
        "pads": pads,
        "value": value,
        "outputDtype": output_dtype,
    }

valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32))
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
gradient = bg.func.grad(
    lambda value: (F.pad(value, tuple(valid["pad"]), value=valid["value"]) * cotangent).sum()
)(source)

vmap_spec = fixture["vmap"]
mapped = bg.func.vmap(
    lambda row: F.pad(row, tuple(vmap_spec["pad"]))
)(bg.from_numpy(np.asarray(vmap_spec["inputValues"], dtype=np.float32)))

padded = F.pad(source, tuple(valid["pad"]), value=valid["value"])
onnx = parse_pad(bg.onnx.export_inference(padded, input_buffers=(source,)))
bool_source = bg.from_numpy(np.asarray([True], dtype=np.bool_))

{
    "registered": get_rule(OP_PAD) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedArg": dict(mapped._uop.arg),
    "onnx": onnx,
    "boolOnnxError": error(lambda: bg.onnx.export_inference(
        F.pad(bool_source, (1, 1)), input_buffers=(bool_source,)
    )),
    "planError": error(lambda: bg.gpu_plan_summary(padded)),
    "webgpuSupported": [OP_PAD in supported_opcodes(), OP_SLICE in supported_opcodes()],
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedGradient);
    expect(result.gradientOps).toContain("SLICE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_PAD_CONFORMANCE.vmap.expectedValues);
    expect(result.mappedArg).toEqual({
      pad_width: FRAMEWORK_PAD_CONFORMANCE.vmap.expectedPadWidth,
      mode: "constant",
      value: 0,
    });
    expect(result.onnx).toEqual(FRAMEWORK_PAD_CONFORMANCE.onnx);
    expect(result.boolOnnxError).toMatch(/^OnnxUnmappableOp: .*bool/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*PAD/u);
    expect(result.webgpuSupported).toEqual([false, false]);
  });

  it("rejects malformed or mutated PAD at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_PAD
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.ones((2,), dtype=np.float32))
valid = F.pad(source, (1, 1))
mutated = F.pad(source, (1, 1))
mutated._uop.arg["pad_width"] = ([1, 1],)
dy = bg.from_numpy(np.ones(valid.shape, dtype=np.float32))._uop
large = UOp(OP_BUFFER, (), (1 << 27,), "float32", arg="pad:large")
high_rank = UOp(OP_BUFFER, (), (1,) * 33, "float32", arg="pad:rank")

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "missingInput": error(lambda: UOp(OP_PAD, (), (), "float32", arg={
        "pad_width": (), "mode": "constant", "value": 0.0
    })),
    "wrongFields": error(lambda: UOp(OP_PAD, (source._uop,), (4,), "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_PAD, (source._uop,), (5,), "float32", arg={
        "pad_width": ((1, 1),), "mode": "constant", "value": 0.0
    })),
    "wrongDtype": error(lambda: UOp(OP_PAD, (source._uop,), (4,), "float64", arg={
        "pad_width": ((1, 1),), "mode": "constant", "value": 0.0
    })),
    "wrongMode": error(lambda: UOp(OP_PAD, (source._uop,), (4,), "float32", arg={
        "pad_width": ((1, 1),), "mode": "reflect", "value": 0.0
    })),
    "noncanonicalValue": error(lambda: UOp(OP_PAD, (source._uop,), (4,), "float32", arg={
        "pad_width": ((1, 1),), "mode": "constant", "value": 0
    })),
    "resourceBound": error(lambda: UOp(OP_PAD, (large,), (1 << 27,), "float32", arg={
        "pad_width": ((0, 0),), "mode": "constant", "value": 0.0
    })),
    "rankBound": error(lambda: UOp(OP_PAD, (high_rank,), (1,) * 33, "float32", arg={
        "pad_width": ((0, 0),) * 33, "mode": "constant", "value": 0.0
    })),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_PAD)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_PAD)(
        mutated._uop, {id(source._uop): source._uop}, 1
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated)),
}
`);

    expect(result.missingInput).toMatch(/^ShapeError: .*at least one input/u);
    expect(result.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.wrongShape).toMatch(/^ShapeError: .*output shape/u);
    expect(result.wrongDtype).toMatch(/^ShapeError: .*preserve/u);
    expect(result.wrongMode).toMatch(/^ShapeError: .*mode/u);
    expect(result.noncanonicalValue).toMatch(/^ShapeError: .*canonical/u);
    expect(result.resourceBound).toMatch(/^ShapeError: .*byte ceiling/u);
    expect(result.rankBound).toMatch(/^ShapeError: .*rank ceiling/u);
    expect(result.openArgCpu).toMatch(/^RealizationError: .*exact pair/u);
    expect(result.openArgVjp).toMatch(/^ShapeError: .*exact pair/u);
    expect(result.openArgVmap).toMatch(/^ShapeError: .*exact pair/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError: .*exact pair/u);
    expect(result.openArgPlan).toMatch(/^ShapeError: .*exact pair/u);
  });
});
