import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_INTERPOLATE_2D_CONFORMANCE } from "../../../test-support/framework-interpolate2d-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

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

describe("typed torch.nn.functional.interpolate parity", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares forward, backward, dtype, scale, and ownership semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_INTERPOLATE_2D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def run_case(spec, mode, **kwargs):
    source = grad.Tensor(
        np.asarray(fixture["input"], dtype=np.float32),
        requires_grad=True,
    )
    output = F.interpolate(source, mode=mode, **kwargs)
    values = output.data
    output.backward(grad.Tensor(
        np.asarray(spec["cotangent"], dtype=np.float32)
    ))
    values[0, 0, 0, 0] = np.float32(999.0)
    replay = F.interpolate(source, mode=mode, **kwargs)
    return {
        "values": replay.tolist(),
        "gradient": source.grad.tolist(),
        "ownsData": bool(values.flags["OWNDATA"]),
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
    typed = grad.Tensor(
        np.asarray(fixture["input"], dtype=np.dtype(dtype)),
        dtype=dtype,
    )
    resized = F.interpolate(typed, size=(3, 3), mode="bilinear")
    dtype_cases.append({
        "input": dtype,
        "output": resized.dtype,
        "numpy": resized.data.dtype.name,
    })

non_integral = np.asarray(
    fixture["nonIntegralScale"]["input"],
    dtype=np.float32,
)
without_recompute = F.interpolate(
    grad.Tensor(non_integral),
    scale_factor=tuple(fixture["nonIntegralScale"]["scaleFactor"]),
    mode="bilinear",
    align_corners=False,
    recompute_scale_factor=False,
).data
with_recompute = F.interpolate(
    grad.Tensor(non_integral),
    scale_factor=tuple(fixture["nonIntegralScale"]["scaleFactor"]),
    mode="bilinear",
    align_corners=False,
    recompute_scale_factor=True,
).data

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
    const expectedNonIntegral =
      FRAMEWORK_INTERPOLATE_2D_CONFORMANCE.nonIntegralScale;
    const nonIntegral = result.nonIntegral as Record<string, unknown>;
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

  it("fails closed for unsupported or resource-excess requests", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_INTERPOLATE_2D_CONFORMANCE);
    const result = await target.run<Record<string, string>>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

source = grad.Tensor(np.asarray(fixture["input"], dtype=np.float32))
{
    "nonTensor": error(lambda: F.interpolate([1.0], size=2)),
    "rank": error(lambda: F.interpolate(
        grad.Tensor(np.ones((1, 2, 2), dtype=np.float32)), size=3
    )),
    "dtype": error(lambda: F.interpolate(
        grad.Tensor(
            np.ones((1, 1, 2, 2), dtype=np.int32),
            dtype="int32",
        ),
        size=3,
    )),
    "bothGeometry": error(lambda: F.interpolate(
        source, size=3, scale_factor=2
    )),
    "missingGeometry": error(lambda: F.interpolate(source)),
    "badSize": error(lambda: F.interpolate(source, size=(3, True))),
    "zeroSize": error(lambda: F.interpolate(source, size=(3, 0))),
    "badScale": error(lambda: F.interpolate(
        source, scale_factor=(2, float("nan"))
    )),
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
}
`);

    for (const value of Object.values(result)) {
      expect(value).not.toBe("no_error");
    }
  });
});
