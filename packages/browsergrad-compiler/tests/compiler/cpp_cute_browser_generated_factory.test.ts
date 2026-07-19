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
    .toBe("796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH).toBe(27_125);
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY.constructor.name).toBe("AsyncFunction");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY).toEqual({
    source: "reviewed-clean-validation-build-output",
    packageOwned: true,
    exactSourcePinned: true,
    cleanValidationRunId: "29681845216",
    cleanValidationSourceRevision: "aca7ee4ea799357b2c0ee3a57f6687e2139e7b7b",
    cleanValidationWasmSha256:
      "5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758",
    cleanValidationWasmByteLength: 31_641_377,
    exactCleanFactoryMatch: true,
    cleanBuildVerified: true,
    reproducibilityVerified: false,
    workerBundleVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(Object.isFrozen(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY)).toBe(true);
});
