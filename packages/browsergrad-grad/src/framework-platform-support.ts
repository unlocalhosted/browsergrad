import pkg from "../package.json" with { type: "json" };

import { GRAD_FRAMEWORK_PLATFORM_OPERATIONS } from "./framework-platform-support.generated.js";

export const GRAD_FRAMEWORK_ID = "browsergrad.grad" as const;
export const GRAD_FRAMEWORK_VERSION = pkg.version;
export const GRAD_FRAMEWORK_CONTRACT_SCHEMA =
  "browsergrad.grad.compatibility.v1" as const;
export const GRAD_FRAMEWORK_CONTRACT_VERSION =
  Object.freeze({ major: 2, minor: 0 }) as Readonly<{
    major: 2;
    minor: 0;
  }>;

export const GRAD_FRAMEWORK_DECISION_FIELDS = Object.freeze([
  "cpu",
  "closureAutograd",
  "symbolicVjp",
  "functionalGrad",
  "vmap",
  "onnxExport",
  "tensorPlan",
  "webgpu",
  "residency",
  "materialization",
] as const);

export type GradFrameworkDecisionField =
  (typeof GRAD_FRAMEWORK_DECISION_FIELDS)[number];

export type GradFrameworkOperationDecisions = Readonly<
  Record<GradFrameworkDecisionField, string>
>;

export interface GradFrameworkPlatformOperation {
  readonly operationId: string;
  readonly publicSurface: string;
  readonly implementationId: string;
  readonly semanticState: "verified-eager-contract";
  readonly shapeContract: string;
  readonly dtypeContract: string;
  readonly decisions: GradFrameworkOperationDecisions;
}

export interface GradFrameworkPlatformSupportSource {
  readonly frameworkId: typeof GRAD_FRAMEWORK_ID;
  readonly frameworkVersion: string;
  readonly contractSchema: typeof GRAD_FRAMEWORK_CONTRACT_SCHEMA;
  readonly contractVersion: typeof GRAD_FRAMEWORK_CONTRACT_VERSION;
  readonly operations: readonly GradFrameworkPlatformOperation[];
}

export function frameworkPlatformSupportSource():
  GradFrameworkPlatformSupportSource {
  return {
    frameworkId: GRAD_FRAMEWORK_ID,
    frameworkVersion: GRAD_FRAMEWORK_VERSION,
    contractSchema: GRAD_FRAMEWORK_CONTRACT_SCHEMA,
    contractVersion: { ...GRAD_FRAMEWORK_CONTRACT_VERSION },
    operations: GRAD_FRAMEWORK_PLATFORM_OPERATIONS.map((operation) => ({
      ...operation,
      decisions: { ...operation.decisions },
    })),
  };
}
