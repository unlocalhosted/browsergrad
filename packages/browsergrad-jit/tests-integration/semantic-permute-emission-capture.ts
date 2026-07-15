import { DENSE_PERMUTATION_VIEW_COPY_FIXTURES } from "../../../test-support/dense-permutation-view-copy-fixtures";
import { clearNamespace, getJitTarget } from "./pyodide-host";

export interface JitSemanticPermuteCapture {
  readonly schema: "browsergrad.jit.semantic-permute-emission-capture";
  readonly version: Readonly<{ readonly major: 1; readonly minor: 0 }>;
  readonly cases: readonly Readonly<{
    readonly caseId: string;
    readonly plan: unknown;
    /** Exact canonical JSON string produced by the production JIT seam. */
    readonly semanticRequestsJson: string;
  }>[];
}

/**
 * Capture the exact production JIT submission consumed by the real-device
 * browser lane. No plan IDs, shapes, axes, or request fields are reconstructed
 * in TypeScript.
 */
export async function captureJitSemanticPermuteSubmissions(): Promise<string> {
  const target = await getJitTarget();
  await clearNamespace(target);
  const fixturesJson = JSON.stringify(
    DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map((fixture) => ({
      id: fixture.id,
      inputShape: fixture.request.inputShape,
      axes: fixture.request.axes,
    })),
  );
  return target.run<string>(`
import browsergrad_jit as bg
import json
from browsergrad_jit._realize_webgpu import _tensor_plan_submission
from browsergrad_jit._tensor_proxy import _from_buffer_id

fixtures = json.loads(${JSON.stringify(fixturesJson)})
captures = []
for fixture in fixtures:
    shape = tuple(int(extent) for extent in fixture["inputShape"])
    tensor = _from_buffer_id(
        "semantic-permute:" + fixture["id"],
        shape,
        "float32",
        session=bg.get_default_session(),
    )
    output = tensor.permute(*fixture["axes"])
    plan, _semantic_requests, semantic_requests_json = _tensor_plan_submission(output._uop)
    captures.append({
        "caseId": fixture["id"],
        "plan": plan,
        "semanticRequestsJson": semantic_requests_json,
    })

json.dumps({
    "schema": "browsergrad.jit.semantic-permute-emission-capture",
    "version": {"major": 1, "minor": 0},
    "cases": captures,
}, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
`);
}
