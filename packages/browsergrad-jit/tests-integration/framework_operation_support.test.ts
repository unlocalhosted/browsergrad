import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

const FRAMEWORK_SUPPORT = {
  schema: "browsergrad.jit.framework-operation-contracts",
  version: { major: 1, minor: 0 },
  operations: [
    {
      contractId: "browsergrad.jit.framework.tensor.abs.v1",
      publicSurface: "Tensor.abs",
      opcode: "ABS",
      semanticState: "typed",
      shapeContract: "preserve-unary-input",
      dtypeContract: "preserve-real-numeric-input",
      decisions: {
        cpu: "supported-numpy-dtype-preserving",
        closureAutograd: "supported-sign-derivative",
        symbolicVjp: "supported-sign-derivative",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-direct-unary-export-dtypes",
        tensorPlan: "refused-no-portable-lowering",
        webgpu: "refused-no-tensor-plan-kernel",
        residency: "host-materialized",
        materialization: "cpu-owning-array",
      },
      retiredOpaqueOperationId: "jit.custom.abs.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.clamp.v1",
      publicSurface: "Tensor.clamp",
      opcode: "CLAMP",
      semanticState: "typed",
      shapeContract: "preserve-unary-input",
      dtypeContract: "preserve-floating-input",
      decisions: {
        cpu: "supported-numpy-dtype-preserving",
        closureAutograd: "supported-inclusive-bound-mask",
        symbolicVjp: "supported-inclusive-bound-mask",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-clip-export-dtypes",
        tensorPlan: "refused-no-portable-lowering",
        webgpu: "refused-no-tensor-plan-kernel",
        residency: "host-materialized",
        materialization: "cpu-owning-array",
      },
      retiredOpaqueOperationId: "jit.custom.clamp.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.cos.v1",
      publicSurface: "Tensor.cos",
      opcode: "COS",
      semanticState: "typed",
      shapeContract: "preserve-unary-input",
      dtypeContract: "preserve-floating-input",
      decisions: {
        cpu: "supported-numpy-dtype-preserving",
        closureAutograd: "supported-negative-sin-derivative",
        symbolicVjp: "supported-negative-sin-derivative",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-direct-unary-export-dtypes",
        tensorPlan: "refused-no-portable-lowering",
        webgpu: "refused-no-tensor-plan-kernel",
        residency: "host-materialized",
        materialization: "cpu-owning-array",
      },
      retiredOpaqueOperationId: "jit.custom.cos.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.expand.v1",
      publicSurface: "Tensor.expand",
      opcode: "BROADCAST_TO",
      semanticState: "typed",
      shapeContract: "static-broadcast-with-existing-dim-minus-one",
      dtypeContract: "preserve-input",
      decisions: {
        cpu: "supported-numpy-owning-copy",
        closureAutograd: "supported-unbroadcast-sum",
        symbolicVjp: "supported-unbroadcast-sum",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-expand",
        tensorPlan: "supported-primitive",
        webgpu: "profile-nonempty-f32-rank-at-most-4",
        residency: "supported-materializing-and-resident",
        materialization: "cpu-owning-copy",
      },
      retiredOpaqueOperationId: "jit.custom.expand.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.flip.v1",
      publicSurface: "Tensor.flip",
      opcode: "FLIP",
      semanticState: "typed",
      shapeContract: "preserve-single-axis-reverse",
      dtypeContract: "preserve-input",
      decisions: {
        cpu: "supported-numpy-owning-copy",
        closureAutograd: "supported-involutive-flip",
        symbolicVjp: "supported-involutive-flip",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis-with-axis-shift",
        onnxExport: "supported-opset17-slice-float32-int32-int64-bool",
        tensorPlan: "refused-negative-stride-profile",
        webgpu: "refused-negative-stride-profile",
        residency: "host-materialized",
        materialization: "cpu-owning-copy",
      },
      retiredOpaqueOperationId: "jit.custom.flip.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.repeat.v1",
      publicSurface: "Tensor.repeat",
      opcode: "REPEAT",
      semanticState: "typed",
      shapeContract: "tile-multipliers-with-left-rank-padding",
      dtypeContract: "preserve-input",
      decisions: {
        cpu: "supported-numpy-owning-copy",
        closureAutograd: "supported-tile-block-sum",
        symbolicVjp: "supported-tile-block-sum",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis-with-unit-repeat",
        onnxExport: "supported-opset17-tile-float32-int32-int64-bool",
        tensorPlan: "refused-no-canonical-tile-layout-profile",
        webgpu: "refused-no-canonical-tile-layout-profile",
        residency: "host-materialized",
        materialization: "cpu-owning-copy",
      },
      retiredOpaqueOperationId: "jit.custom.repeat.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.repeat-interleave.v1",
      publicSurface: "Tensor.repeat_interleave",
      opcode: "REPEAT_INTERLEAVE",
      semanticState: "typed",
      shapeContract: "selected-axis-times-repeat-count",
      dtypeContract: "preserve-input",
      decisions: {
        cpu: "supported-numpy-owning-copy",
        closureAutograd: "supported-selected-axis-block-sum",
        symbolicVjp: "supported-selected-axis-block-sum",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis-with-axis-shift",
        onnxExport: "supported-opset17-unsqueeze-tile-reshape-float32-int32-int64-bool",
        tensorPlan: "refused-no-canonical-selected-axis-replication-profile",
        webgpu: "refused-no-canonical-selected-axis-replication-profile",
        residency: "host-materialized",
        materialization: "cpu-owning-copy",
      },
      retiredOpaqueOperationId: "jit.custom.repeat-interleave.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.sign.v1",
      publicSurface: "Tensor.sign",
      opcode: "SIGN",
      semanticState: "typed",
      shapeContract: "preserve-unary-input",
      dtypeContract: "preserve-real-numeric-input",
      decisions: {
        cpu: "supported-numpy-dtype-preserving",
        closureAutograd: "supported-zero-derivative",
        symbolicVjp: "supported-zero-derivative",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-direct-unary-export-dtypes",
        tensorPlan: "refused-no-portable-lowering",
        webgpu: "refused-no-tensor-plan-kernel",
        residency: "host-materialized",
        materialization: "cpu-owning-array",
      },
      retiredOpaqueOperationId: "jit.custom.sign.v0",
    },
    {
      contractId: "browsergrad.jit.framework.tensor.sin.v1",
      publicSurface: "Tensor.sin",
      opcode: "SIN",
      semanticState: "typed",
      shapeContract: "preserve-unary-input",
      dtypeContract: "preserve-floating-input",
      decisions: {
        cpu: "supported-numpy-dtype-preserving",
        closureAutograd: "supported-cos-derivative",
        symbolicVjp: "supported-cos-derivative",
        functionalGrad: "supported-via-symbolic-vjp",
        vmap: "supported-leading-batch-axis",
        onnxExport: "supported-opset17-direct-unary-export-dtypes",
        tensorPlan: "refused-no-portable-lowering",
        webgpu: "refused-no-tensor-plan-kernel",
        residency: "host-materialized",
        materialization: "cpu-owning-array",
      },
      retiredOpaqueOperationId: "jit.custom.sin.v0",
    },
  ].sort((left, right) => left.contractId < right.contractId ? -1 : left.contractId > right.contractId ? 1 : 0),
};

describe("Gate 6 executable framework-operation support", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("reports detached public decisions from the contract used to validate execution", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      first: typeof FRAMEWORK_SUPPORT;
      second: typeof FRAMEWORK_SUPPORT;
      validatedContractId: string;
      normalizedShape: number[];
      values: number[][];
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._framework_contracts import validate_framework_operation_contract

expanded = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32)).expand(-1, 3)
record, normalized = validate_framework_operation_contract(expanded._uop)
first = bg.framework_operation_support()
first["operations"][0]["decisions"]["cpu"] = "forged"
first["operations"].append({"contractId": "forged"})
second = bg.framework_operation_support()

{
    "first": first,
    "second": second,
    "validatedContractId": record.contract_id,
    "normalizedShape": list(normalized),
    "values": expanded.numpy().tolist(),
}
`);

    expect(result.first.operations).toHaveLength(10);
    expect(result.first.operations[0]?.decisions.cpu).toBe("forged");
    expect(result.second).toEqual(FRAMEWORK_SUPPORT);
    expect(result.validatedContractId).toBe("browsergrad.jit.framework.tensor.expand.v1");
    expect(result.normalizedShape).toEqual([2, 3]);
    expect(result.values).toEqual([[1, 1, 1], [2, 2, 2]]);
  });

  it("rejects duplicate, oversized, mutable, open, and unregistered registry inputs", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import copy
import json
from browsergrad_jit._framework_contracts import _parse_registry_payload

valid = bg.framework_operation_support()

def error(payload):
    try:
        _parse_registry_payload(payload)
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

extra = copy.deepcopy(valid)
extra["operations"][0]["backendHint"] = "webgpu"
unknown_enum = copy.deepcopy(valid)
unknown_enum["operations"][0]["decisions"]["cpu"] = "supported-somehow"
empty = copy.deepcopy(valid)
empty["operations"] = []
boolean_version = copy.deepcopy(valid)
boolean_version["version"]["major"] = True

{
    "duplicate": error(b'{"schema":"a","schema":"b"}'),
    "oversized": error(b"x" * (16 * 1024 + 1)),
    "mutable": error(bytearray(b"{}")),
    "extra": error(json.dumps(extra).encode("utf-8")),
    "unknownEnum": error(json.dumps(unknown_enum).encode("utf-8")),
    "empty": error(json.dumps(empty).encode("utf-8")),
    "booleanVersion": error(json.dumps(boolean_version).encode("utf-8")),
}
`);

    expect(errors.duplicate).toMatch(/^ValueError: .*duplicates field/u);
    expect(errors.oversized).toMatch(/^ValueError: .*1\.\.16384 bytes/u);
    expect(errors.mutable).toMatch(/^ValueError: .*immutable bytes/u);
    expect(errors.extra).toMatch(/^ValueError: .*fields changed/u);
    expect(errors.unknownEnum).toMatch(/^ValueError: .*not registered/u);
    expect(errors.empty).toMatch(/^ValueError: .*non-empty list/u);
    expect(errors.booleanVersion).toMatch(/^ValueError: .*exactly 1\.0/u);
  });
});
