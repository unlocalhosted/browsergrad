import { expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_GENERATED_FACTORY,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_ID,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
} from "../../src/cpp_cute_browser_generated_factory.js";

it("pins the exact package-owned generated factory without widening its authority", () => {
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_ID).toBe(
    `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`,
  );
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256)
    .toBe("ef6b757b053fe6ce232b7f32bb6d7e747211ad65102940607e8346fd027f63c3");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH).toBe(23_916);
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY.constructor.name).toBe("AsyncFunction");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY).toEqual({
    source: "reviewed-clean-validation-build-output",
    packageOwned: true,
    exactSourcePinned: true,
    cleanValidationRunId: "29674887505",
    cleanValidationSourceRevision: "d631cc3223777cada1284cf5284b1a0efdcb36c4",
    cleanValidationWasmSha256:
      "b7a5daf6d121c306a2d07b5d3c14c00a664aaa2ff4ae3357a8b389326eeeb06f",
    cleanValidationWasmByteLength: 31_307_826,
    exactCleanFactoryMatch: true,
    cleanBuildVerified: true,
    reproducibilityVerified: false,
    workerBundleVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(Object.isFrozen(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY)).toBe(true);
});
