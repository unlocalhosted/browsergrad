import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_INTERPOLATE_2D_CONFORMANCE } from "../../../test-support/framework-interpolate2d-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectClose(actual: unknown, expected: unknown, digits = 5): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expect(actual as unknown[]).toHaveLength(expected.length);
    expected.forEach((value, index) => {
      expectClose((actual as unknown[])[index], value, digits);
    });
    return;
  }
  expect(actual as number).toBeCloseTo(expected as number, digits);
}

describe("Gate 6 typed torch.nn.functional.interpolate contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("preserves dtype, owns output storage, and matches forward and closure VJP", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_INTERPOLATE_2D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def run_case(spec, mode, **kwargs):
    source = bg.from_numpy(
        np.asarray(fixture["input"], dtype=np.float32),
        requires_grad=True,
    )
    output = F.interpolate(source, mode=mode, **kwargs)
    first = output.numpy()
    output.backward(bg.from_numpy(
        np.asarray(spec["cotangent"], dtype=np.float32)
    ))
    first[0, 0, 0, 0] = np.float32(999.0)
    return {
        "op": output._uop.op,
        "arg": dict(output._uop.arg),
        "values": F.interpolate(
            source,
            mode=mode,
            **kwargs,
        ).numpy().tolist(),
        "gradient": source.grad.numpy().tolist(),
        "ownsData": bool(first.flags["OWNDATA"]),
    }

nearest = run_case(
    fixture["nearest"],
    "nearest",
    size=tuple(fixture["nearest"]["size"]),
)
bilinear = run_case(
    fixture["bilinear"],
    "bilinear",
    scale_factor=fixture["bilinear"]["scaleFactor"],
    align_corners=False,
)
aligned = run_case(
    fixture["alignCorners"],
    "bilinear",
    size=tuple(fixture["alignCorners"]["size"]),
    align_corners=True,
)

dtype_cases = []
for dtype in ("float16", "float32", "float64"):
    typed = bg.from_numpy(np.asarray(fixture["input"], dtype=np.dtype(dtype)))
    resized = F.interpolate(typed, size=(3, 3), mode="bilinear")
    dtype_cases.append({
        "input": dtype,
        "output": resized.dtype,
        "numpy": resized.numpy().dtype.name,
    })

non_integral = np.asarray(
    fixture["nonIntegralScale"]["input"],
    dtype=np.float32,
)
without_recompute = F.interpolate(
    bg.from_numpy(non_integral),
    scale_factor=tuple(fixture["nonIntegralScale"]["scaleFactor"]),
    mode="bilinear",
    align_corners=False,
    recompute_scale_factor=False,
).numpy()
with_recompute = F.interpolate(
    bg.from_numpy(non_integral),
    scale_factor=tuple(fixture["nonIntegralScale"]["scaleFactor"]),
    mode="bilinear",
    align_corners=False,
    recompute_scale_factor=True,
).numpy()

{
    "nearest": nearest,
    "bilinear": bilinear,
    "aligned": aligned,
    "dtypeCases": dtype_cases,
    "nonIntegral": {
        "shape": list(without_recompute.shape),
        "withoutFirst": without_recompute[0, 0, 0].tolist(),
        "withoutLast": without_recompute[0, 0, -1].tolist(),
        "withFirst": with_recompute[0, 0, 0].tolist(),
        "withLast": with_recompute[0, 0, -1].tolist(),
    },
}
`);

    const nearest = result.nearest as Record<string, unknown>;
    expect(nearest.op).toBe("INTERPOLATE_2D");
    expect(nearest.arg).toEqual({
      align_corners: false,
      batch_rank: 0,
      mode: "nearest",
      output_size: FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nearest.size,
      recompute_scale_factor: false,
      scale_factors: undefined,
    });
    expect(nearest.values).toEqual(
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nearest.expected,
    );
    expectClose(
      nearest.gradient,
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nearest.gradient,
    );
    expect(nearest.ownsData).toBe(true);

    const bilinear = result.bilinear as Record<string, unknown>;
    expectClose(
      bilinear.values,
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.bilinear.expected,
    );
    expectClose(
      bilinear.gradient,
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.bilinear.gradient,
    );
    const aligned = result.aligned as Record<string, unknown>;
    expectClose(
      aligned.values,
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.alignCorners.expected,
    );
    expectClose(
      aligned.gradient,
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.alignCorners.gradient,
    );
    expect(result.dtypeCases).toEqual([
      { input: "float16", output: "float16", numpy: "float16" },
      { input: "float32", output: "float32", numpy: "float32" },
      { input: "float64", output: "float64", numpy: "float64" },
    ]);

    const nonIntegral = result.nonIntegral as Record<string, unknown>;
    const expectedNonIntegral =
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nonIntegralScale;
    expect(nonIntegral.shape).toEqual(expectedNonIntegral.expectedShape);
    expectClose(
      nonIntegral.withoutFirst,
      expectedNonIntegral.withoutRecomputeFirstRow,
    );
    expectClose(
      nonIntegral.withoutLast,
      expectedNonIntegral.withoutRecomputeLastRow,
    );
    expectClose(
      nonIntegral.withFirst,
      expectedNonIntegral.withRecomputeFirstRow,
    );
    expectClose(
      nonIntegral.withLast,
      expectedNonIntegral.withRecomputeLastRow,
    );
  });

  it("supports symbolic VJP, vmap, checkpoint replay, ONNX Resize, and explicit GPU refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_INTERPOLATE_2D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
import struct
from browsergrad_jit._ir import (
    OP_INTERPOLATE_2D,
    OP_INTERPOLATE_2D_VJP,
    toposort,
)
from browsergrad_jit._vjp import get_rule
from browsergrad_jit.utils.checkpoint import checkpoint

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

source_values = np.asarray(fixture["input"], dtype=np.float32)
source = bg.from_numpy(source_values)
symbolic = bg.func.grad(
    lambda value: F.interpolate(
        value,
        size=tuple(fixture["nearest"]["size"]),
        mode="nearest",
    ).sum()
)(source)
batched_values = np.stack((source_values, source_values + 10.0), axis=0)
batched = bg.from_numpy(batched_values)
mapped = bg.func.vmap(
    lambda value: F.interpolate(
        value,
        size=tuple(fixture["nearest"]["size"]),
        mode="nearest",
    )
)(batched)
mapped_gradient = bg.func.vmap(bg.func.grad(
    lambda value: F.interpolate(
        value,
        size=tuple(fixture["nearest"]["size"]),
        mode="nearest",
    ).sum()
))(batched)

checkpoint_source = bg.from_numpy(source_values, requires_grad=True)
checkpointed = checkpoint(
    lambda value: F.interpolate(
        value,
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    ),
    checkpoint_source,
)
checkpoint_first = checkpointed.numpy()
checkpoint_second = checkpointed.numpy()
checkpointed.sum().backward()

nearest = F.interpolate(source, size=(3, 5), mode="nearest")
bilinear = F.interpolate(
    source,
    scale_factor=2,
    mode="bilinear",
    align_corners=False,
)

${ONNX_PROTOBUF_TEST_HELPERS}

def resize_node(model):
    graph = next(
        payload for number, wire, payload in fields(model)
        if number == 7 and wire == 2
    )
    initializers = {}
    for number, wire, payload in fields(graph):
        if number != 5 or wire != 2:
            continue
        tensor_fields = list(fields(payload))
        name = next(
            item.decode("utf-8")
            for field, kind, item in tensor_fields
            if field == 8 and kind == 2
        )
        raw = next(
            (
                item for field, kind, item in tensor_fields
                if field == 9 and kind == 2
            ),
            b"",
        )
        dtype = next(
            item for field, kind, item in tensor_fields
            if field == 2 and kind == 0
        )
        if "interpolate_scales" in name:
            initializers["scales"] = [
                value[0] for value in struct.iter_unpack("<f", raw)
            ]
            initializers["scalesDtype"] = dtype
        elif "interpolate_sizes" in name:
            initializers["sizes"] = [
                value[0] for value in struct.iter_unpack("<q", raw)
            ]
            initializers["sizesDtype"] = dtype
    for number, wire, payload in fields(graph):
        if number != 1 or wire != 2:
            continue
        node_fields = list(fields(payload))
        op_type = next(
            item.decode("utf-8") for field, kind, item in node_fields
            if field == 4 and kind == 2
        )
        if op_type != "Resize":
            continue
        attributes = {}
        for field, kind, attribute in node_fields:
            if field != 5 or kind != 2:
                continue
            attribute_fields = list(fields(attribute))
            name = next(
                item.decode("utf-8")
                for child, child_kind, item in attribute_fields
                if child == 1 and child_kind == 2
            )
            value = next(
                item.decode("utf-8")
                for child, child_kind, item in attribute_fields
                if child == 4 and child_kind == 2
            )
            attributes[name] = value
        return {
            "inputCount": sum(
                1 for field, kind, _ in node_fields
                if field == 1 and kind == 2
            ),
            "attributes": attributes,
            "initializers": initializers,
        }
    raise RuntimeError("Resize node missing")

{
    "registered": get_rule(OP_INTERPOLATE_2D) is not None,
    "symbolicOps": [node.op for node in toposort(symbolic._uop)],
    "symbolic": symbolic.numpy().tolist(),
    "mappedShape": list(mapped.shape),
    "mapped": mapped.numpy().tolist(),
    "mappedGradient": mapped_gradient.numpy().tolist(),
    "checkpointReplay": bool(np.array_equal(
        checkpoint_first, checkpoint_second
    )),
    "checkpointGradientFinite": bool(
        np.isfinite(checkpoint_source.grad.numpy()).all()
    ),
    "nearestOnnx": resize_node(
        bg.onnx.export_inference(nearest, input_buffers=(source,))
    ),
    "bilinearOnnx": resize_node(
        bg.onnx.export_inference(bilinear, input_buffers=(source,))
    ),
    "planError": error(lambda: bg.gpu_plan_summary(nearest)),
    "vjpOnnxError": error(lambda: bg.onnx.export_inference(
        symbolic, input_buffers=(source,)
    )),
}
`);

    expect(result.registered).toBe(true);
    expect(result.symbolicOps).toContain("INTERPOLATE_2D_VJP");
    expect(result.symbolic).toEqual([[[[6, 4], [3, 2]]]]);
    expect(result.mappedShape).toEqual([2, 1, 1, 3, 5]);
    expect((result.mapped as unknown[])[0]).toEqual(
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nearest.expected,
    );
    expect(result.mappedGradient).toEqual([
      [[[[6, 4], [3, 2]]]],
      [[[[6, 4], [3, 2]]]],
    ]);
    expect(result.checkpointReplay).toBe(true);
    expect(result.checkpointGradientFinite).toBe(true);
    expect(result.nearestOnnx).toEqual({
      inputCount: 4,
      attributes: {
        coordinate_transformation_mode: "asymmetric",
        mode: "nearest",
        nearest_mode: "floor",
      },
      initializers: {
        scales: [],
        scalesDtype: 1,
        sizes: [1, 1, 3, 5],
        sizesDtype: 7,
      },
    });
    expect(result.bilinearOnnx).toEqual({
      inputCount: 4,
      attributes: {
        coordinate_transformation_mode: "half_pixel",
        mode: "linear",
      },
      initializers: {
        scales: [1, 1, 2, 2],
        scalesDtype: 1,
        sizes: [],
        sizesDtype: 7,
      },
    });
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*INTERPOLATE_2D/u);
    expect(result.vjpOnnxError).toMatch(/^OnnxUnmappableOp: .*INTERPOLATE_2D_VJP/u);
  });

  it("fails closed for invalid requests, resource excess, and mutated typed IR", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_INTERPOLATE_2D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._framework_contracts import (
    INTERPOLATE_2D_BILINEAR_WORK_VISIT_FACTOR,
    INTERPOLATE_2D_NEAREST_WORK_VISIT_FACTOR,
    INTERPOLATE_2D_OUTPUT_BYTE_MAX,
    INTERPOLATE_2D_OUTPUT_EXTENT_MAX,
    INTERPOLATE_2D_WORK_ELEMENT_MAX,
    INTERPOLATE_2D_WORKSPACE_BYTE_MAX,
)

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

source = bg.from_numpy(np.asarray(fixture["input"], dtype=np.float32))
valid = F.interpolate(source, size=(3, 5), mode="nearest")
original_size = valid._uop.arg["output_size"]
valid._uop.arg["output_size"] = (3, 4)
mutation_error = error(valid.numpy)
valid._uop.arg["output_size"] = original_size

errors = {
    "nonTensor": error(lambda: F.interpolate([1.0], size=2)),
    "rank": error(lambda: F.interpolate(
        bg.from_numpy(np.ones((1, 2, 2), dtype=np.float32)), size=3
    )),
    "dtype": error(lambda: F.interpolate(
        bg.from_numpy(np.ones((1, 1, 2, 2), dtype=np.int32)), size=3
    )),
    "bothGeometry": error(lambda: F.interpolate(
        source, size=3, scale_factor=2
    )),
    "missingGeometry": error(lambda: F.interpolate(source)),
    "badSize": error(lambda: F.interpolate(source, size=(3, True))),
    "zeroSize": error(lambda: F.interpolate(source, size=(3, 0))),
    "badScale": error(lambda: F.interpolate(source, scale_factor=(2, float("nan")))),
    "tinyScale": error(lambda: F.interpolate(source, scale_factor=0.1)),
    "mode": error(lambda: F.interpolate(source, size=3, mode="bicubic")),
    "nearestAlignment": error(lambda: F.interpolate(
        source, size=3, mode="nearest", align_corners=False
    )),
    "alignmentType": error(lambda: F.interpolate(
        source, size=3, mode="bilinear", align_corners=0
    )),
    "recomputeWithSize": error(lambda: F.interpolate(
        source, size=3, recompute_scale_factor=False
    )),
    "recomputeType": error(lambda: F.interpolate(
        source, scale_factor=2, recompute_scale_factor=1
    )),
    "antialias": error(lambda: F.interpolate(
        source, size=3, mode="bilinear", antialias=True
    )),
    "oversizedExtent": error(lambda: F.interpolate(
        source,
        size=(3, fixture["limits"]["outputExtent"] + 1),
    )),
    "work": error(lambda: F.interpolate(
        source,
        size=(fixture["limits"]["outputExtent"], 9),
        mode="bilinear",
    )),
    "mutation": mutation_error,
}

{
    "errors": errors,
    "limits": {
        "outputBytes": INTERPOLATE_2D_OUTPUT_BYTE_MAX,
        "outputExtent": INTERPOLATE_2D_OUTPUT_EXTENT_MAX,
        "workElements": INTERPOLATE_2D_WORK_ELEMENT_MAX,
        "workspaceBytes": INTERPOLATE_2D_WORKSPACE_BYTE_MAX,
        "nearestWorkVisitFactor": INTERPOLATE_2D_NEAREST_WORK_VISIT_FACTOR,
        "bilinearWorkVisitFactor": INTERPOLATE_2D_BILINEAR_WORK_VISIT_FACTOR,
    },
}
`);

    expect(result.limits).toEqual(
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.limits,
    );
    for (const value of Object.values(result.errors as Record<string, string>)) {
      expect(value).not.toBe("no_error");
    }
  });
});
