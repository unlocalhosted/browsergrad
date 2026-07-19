import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderPackPipelineError,
  parseCppCuteBrowserHeaderPackPipelineArguments,
} from "./cpp_cute_browser_header_pack_pipeline.mjs";

describe("exact header-pack pipeline", () => {
  it("parses one no-serialization pipeline invocation", () => {
    const parsed = parseCppCuteBrowserHeaderPackPipelineArguments([
      "--",
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--output-root=/private/tmp/browsergrad-header-sources",
      "--pack-output-root=/private/tmp/browsergrad-header-packs",
    ]);

    expect(parsed.archives).toHaveLength(7);
    expect(parsed.bsdtarPath).toBe("/usr/bin/bsdtar");
    expect(parsed.sourceOutputRoot).toBe("/private/tmp/browsergrad-header-sources");
    expect(parsed.packOutputRoot).toBe("/private/tmp/browsergrad-header-packs");
    expect(() => parseCppCuteBrowserHeaderPackPipelineArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--output-root=/private/tmp/sources",
    ])).toThrow(CppCuteBrowserHeaderPackPipelineError);
    expect(() => parseCppCuteBrowserHeaderPackPipelineArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--output-root=/private/tmp/sources",
      "--pack-output-root=/private/tmp/one",
      "--pack-output-root=/private/tmp/two",
    ])).toThrow(CppCuteBrowserHeaderPackPipelineError);
  });
});

function archiveArguments(): string[] {
  return [
    "--cuda-cccl-linux-x86-64=/private/tmp/cuda-cccl.tar.xz",
    "--cuda-cudart-linux-x86-64=/private/tmp/cuda-cudart.tar.xz",
    "--cuda-nvcc-linux-x86-64=/private/tmp/cuda-nvcc.tar.xz",
    "--cutlass=/private/tmp/cutlass.tar.gz",
    "--llvm-project=/private/tmp/llvm-project.tar.xz",
    "--ubuntu-noble-libc6-dev-amd64-cross=/private/tmp/libc6-dev.deb",
    "--ubuntu-noble-linux-libc-dev-amd64-cross=/private/tmp/linux-libc-dev.deb",
  ];
}
