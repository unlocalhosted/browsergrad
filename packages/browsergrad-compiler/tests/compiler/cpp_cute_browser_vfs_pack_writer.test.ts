import { describe, expect, it } from "vitest";
import {
  CppCuteBrowserVfsPackError,
  canonicalInspectedCppCuteBrowserVfsPackBytes,
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
  unwrapInspectedCppCuteBrowserVfsPack,
} from "../../src/cpp_cute_browser_vfs_pack.js";

const ENCODER = new TextEncoder();

function file(virtualPath: string, contents: string): { virtualPath: string; bytes: Uint8Array } {
  return { virtualPath, bytes: ENCODER.encode(contents) };
}

async function expectWriterError(
  operation: Promise<unknown>,
  code: CppCuteBrowserVfsPackError["code"],
  path: string,
): Promise<void> {
  await expect(operation).rejects.toMatchObject({ code, path });
}

describe("C++/CuTe canonical browser VFS-pack writer", () => {
  it("encodes deterministic canonical bytes independent of caller order", async () => {
    const inputs = [
      file("cute/layout.hpp", "#pragma once\n"),
      file("clang/22/include/stddef.h", "using size_t = __SIZE_TYPE__;\n"),
      file("empty", ""),
    ];
    const first = await encodeCppCuteBrowserVfsPack(inputs);
    const second = await encodeCppCuteBrowserVfsPack([...inputs].reverse());
    expect(first).toEqual(second);

    const inspected = await inspectCppCuteBrowserVfsPack(first);
    expect(unwrapInspectedCppCuteBrowserVfsPack(inspected).entries.map((entry) => entry.virtualPath)).toEqual([
      "clang/22/include/stddef.h",
      "cute/layout.hpp",
      "empty",
    ]);
    expect(canonicalInspectedCppCuteBrowserVfsPackBytes(inspected)).toEqual(first);
  });

  it("snapshots every input before asynchronous hashing", async () => {
    const inputs = [file("a.h", "first"), file("b.h", "second")];
    const expected = await encodeCppCuteBrowserVfsPack(inputs);
    const pending = encodeCppCuteBrowserVfsPack(inputs);
    inputs[0]!.bytes.fill(0);
    inputs[1]!.bytes.fill(0);
    expect(await pending).toEqual(expected);
  });

  it("rejects duplicate and file-as-parent paths before allocation", async () => {
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("a.h", "one"), file("a.h", "two")]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("dir", "file"), file("dir/child.h", "child")]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([
        file("a.h", "one"),
        { virtualPath: "a.h", bytes: new Proxy(Uint8Array.of(1), {}) },
      ]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([
        file("dir/child.h", "child"),
        { virtualPath: "dir", bytes: new Proxy(Uint8Array.of(1), {}) },
      ]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files",
    );
  });

  it("rejects unsafe paths and enforces independent writer ceilings", async () => {
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("../escape.h", "x")]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files[0].virtualPath",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("large.h", "xx")], { limits: { maxFileBytes: 1 } }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT",
      "$.files[0].bytes",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("a.h", "x")], { limits: { maxPackBytes: 96 } }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT",
      "$.files",
    );
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("a".repeat(4_097), "x")]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT",
      "$.files[0].virtualPath",
    );
  });

  it("rejects sparse/accessor inputs and hostile byte views without invoking accessors", async () => {
    const sparse = new Array(1);
    await expectWriterError(
      encodeCppCuteBrowserVfsPack(sparse),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files[0]",
    );

    let getterCalls = 0;
    const accessor = { virtualPath: "a.h", bytes: Uint8Array.of(1) };
    Object.defineProperty(accessor, "bytes", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return Uint8Array.of(1);
      },
    });
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([accessor]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files[0].bytes",
    );
    expect(getterCalls).toBe(0);

    await expectWriterError(
      encodeCppCuteBrowserVfsPack([{ virtualPath: "a.h", bytes: new Proxy(Uint8Array.of(1), {}) }]),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.files[0].bytes",
    );
    if (typeof SharedArrayBuffer !== "undefined") {
      await expectWriterError(
        encodeCppCuteBrowserVfsPack([{ virtualPath: "a.h", bytes: new Uint8Array(new SharedArrayBuffer(1)) }]),
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
        "$.files[0].bytes",
      );
    }
  });

  it("honors pre-aborted cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    await expectWriterError(
      encodeCppCuteBrowserVfsPack([file("a.h", "x")], { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED",
      "$.signal",
    );
  });

  it("snapshots each file before hostile later-record inspection", async () => {
    const resize = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, "resize")?.value as
      | ((this: ArrayBuffer, byteLength: number) => void)
      | undefined;
    if (resize === undefined) return;

    const ResizableArrayBuffer = ArrayBuffer as unknown as new (
      byteLength: number,
      options: { readonly maxByteLength: number },
    ) => ArrayBuffer;
    const victimBuffer = new ResizableArrayBuffer(2, { maxByteLength: 2 });
    const victimBytes = new Uint8Array(victimBuffer);
    victimBytes.set([1, 2]);
    const hostileLaterFile = new Proxy(file("b.h", "later"), {
      getPrototypeOf(target) {
        resize.call(victimBuffer, 0);
        return Reflect.getPrototypeOf(target);
      },
    });

    const expected = await encodeCppCuteBrowserVfsPack([
      { virtualPath: "a.h", bytes: Uint8Array.of(1, 2) },
      file("b.h", "later"),
    ]);
    await expect(encodeCppCuteBrowserVfsPack([
      { virtualPath: "a.h", bytes: victimBytes },
      hostileLaterFile,
    ])).resolves.toEqual(expected);

    const sameLengthVictim = Uint8Array.of(3, 4);
    const mutatingLaterFile = new Proxy(file("b.h", "later"), {
      getPrototypeOf(target) {
        sameLengthVictim.fill(9);
        return Reflect.getPrototypeOf(target);
      },
    });
    const sameLengthExpected = await encodeCppCuteBrowserVfsPack([
      { virtualPath: "a.h", bytes: Uint8Array.of(3, 4) },
      file("b.h", "later"),
    ]);
    await expect(encodeCppCuteBrowserVfsPack([
      { virtualPath: "a.h", bytes: sameLengthVictim },
      mutatingLaterFile,
    ])).resolves.toEqual(sameLengthExpected);
  });
});
