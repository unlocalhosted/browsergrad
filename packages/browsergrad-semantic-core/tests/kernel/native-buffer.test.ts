import { describe, expect, it } from "vitest";

import {
  copyNativeUint8,
  nativeUint8Slots,
} from "../../src/kernel/native-buffer";

describe("native CPU buffer helpers", () => {
  it("uses the module-captured Uint8Array set intrinsic synchronously", () => {
    const source = new Uint8Array([1, 2, 3, 4]);
    const slots = nativeUint8Slots(source, "$.source");
    const originalSet = Uint8Array.prototype.set;
    let copied: Uint8Array | undefined;
    try {
      Uint8Array.prototype.set = () => {
        throw new Error("poisoned Uint8Array.set");
      };
      copied = copyNativeUint8(source, slots);
    } finally {
      Uint8Array.prototype.set = originalSet;
    }
    expect(copied).toEqual(new Uint8Array([1, 2, 3, 4]));
  });
});
