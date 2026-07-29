import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderDistributionReproducibilityError,
  canonicalCppCuteBrowserHeaderDistributionReproducibilityBytes,
  materializeAndVerifyCppCuteBrowserHeaderDistributionReproducibility,
  parseCppCuteBrowserHeaderDistributionReproducibilityArguments,
  requireCppCuteBrowserHeaderDistributionReproducibilityAuthority,
  verifyCppCuteBrowserHeaderDistributionReproducibility,
} from "./cpp_cute_browser_header_distribution_reproducibility.mjs";

describe("exact header distribution reproducibility", () => {
  it("parses one exact common input closure and four distinct output roots", () => {
    const parsed = parseCppCuteBrowserHeaderDistributionReproducibilityArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--first-source-output-root=/private/tmp/repro/source-a",
      "--first-pack-output-root=/private/tmp/repro/packs-a",
      "--second-source-output-root=/private/tmp/repro/source-b",
      "--second-pack-output-root=/private/tmp/repro/packs-b",
    ]);
    expect(parsed.first.archives).toHaveLength(8);
    expect(parsed.first.sourceOutputRoot).toBe("/private/tmp/repro/source-a");
    expect(parsed.second.packOutputRoot).toBe("/private/tmp/repro/packs-b");
    expect(parsed.first).not.toHaveProperty("allowUnpinnedDiagnosticBsdtar");
    expect(() => parseCppCuteBrowserHeaderDistributionReproducibilityArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--first-source-output-root=/private/tmp/repro/source-a",
      "--first-pack-output-root=/private/tmp/repro/packs-a",
      "--second-source-output-root=/private/tmp/repro/source-b",
      "--second-pack-output-root=/private/tmp/repro/packs-b",
      "--allow-unpinned-diagnostic-bsdtar",
    ])).toThrow(/forbids the unpinned diagnostic archive tool/u);
    expect(() => parseCppCuteBrowserHeaderDistributionReproducibilityArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--first-source-output-root=/private/tmp/repro/one",
      "--first-pack-output-root=/private/tmp/repro/two",
      "--second-source-output-root=/private/tmp/repro/one/child",
      "--second-pack-output-root=/private/tmp/repro/four",
    ])).toThrow(CppCuteBrowserHeaderDistributionReproducibilityError);
  });

  it("rejects forged, copied, or repeated pipeline authorities", async () => {
    const forged = Object.freeze({});
    await expect(verifyCppCuteBrowserHeaderDistributionReproducibility({
      first: forged,
      second: forged,
    } as never)).rejects.toBeInstanceOf(CppCuteBrowserHeaderDistributionReproducibilityError);
    await expect(materializeAndVerifyCppCuteBrowserHeaderDistributionReproducibility({
      first: forged,
      second: forged,
    } as never)).rejects.toBeInstanceOf(CppCuteBrowserHeaderDistributionReproducibilityError);
    expect(() => requireCppCuteBrowserHeaderDistributionReproducibilityAuthority(forged as never))
      .toThrow(CppCuteBrowserHeaderDistributionReproducibilityError);
    expect(() => canonicalCppCuteBrowserHeaderDistributionReproducibilityBytes(forged as never))
      .toThrow(CppCuteBrowserHeaderDistributionReproducibilityError);
  });
});

function archiveArguments(): string[] {
  return [
    "--cuda-cccl-linux-x86-64=/private/tmp/cuda-cccl.tar.xz",
    "--cuda-cudart-linux-x86-64=/private/tmp/cuda-cudart.tar.xz",
    "--cuda-libcurand-linux-x86-64=/private/tmp/cuda-libcurand.tar.xz",
    "--cuda-nvcc-linux-x86-64=/private/tmp/cuda-nvcc.tar.xz",
    "--cutlass=/private/tmp/cutlass.tar.gz",
    "--llvm-project=/private/tmp/llvm-project.tar.xz",
    "--ubuntu-noble-libc6-dev-amd64-cross=/private/tmp/libc6-dev.deb",
    "--ubuntu-noble-linux-libc-dev-amd64-cross=/private/tmp/linux-libc-dev.deb",
  ];
}
