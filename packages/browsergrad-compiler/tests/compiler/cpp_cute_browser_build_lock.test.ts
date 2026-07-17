import {
  canonicalJsonBytes,
  decodeWireJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
  assertCppCuteBrowserBuildReleaseReady,
  canonicalCppCuteBrowserBuildInputLockBytes,
  cppCuteBrowserBuildInputLockResourceBytes,
  cppCuteBrowserBuildReleaseReadiness,
  decodeCppCuteBrowserBuildInputLock,
  deriveCppCuteBrowserBuildInputLockId,
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "../../src/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";

const LOCK_ID =
  "bg.cpp.browser-build-input-lock.sha256.c684dd200e414220b7d8c13aa25d905b3be57cf3dd66a80d49c2b7af9adb3e82";
const RESOURCE_SHA256 = "bea99788f7525e7e25e05446dbc640e4dddbd4a8737ef354675aaf498896c533";
const RECIPE_SHA256 = "fdc8183f42f0e0d66c92961aab51f0c6a3d2129497159c89b8bacb3d9e0a44b4";
const EXTRACTOR_SOURCE_SHA256 = "cae5aa99d17b5ef2d856be02636f5804a11e017ae39acf44ddcaed9643c04593";
const NOTICE_SHA256 = "ae94cc9272e8d3458778dda90db035388450075d5404f736f6daadc7192163d1";
const BLOCKERS = [
  "browsergrad-extractor-artifact-v3",
  "browsergrad-extractor-cuda-dual-pass",
  "browsergrad-extractor-distributed-materialization",
  "browsergrad-extractor-runtime-metrics-export",
  "browsergrad-extractor-source-verification",
  "browsergrad-extractor-vfs-bridge",
  "browsergrad-worker-emscripten-factory-bundle",
  "cuda-header-redistribution",
  "distributed-file-license-manifest",
  "linux-sysroot-redistribution",
  "observed-wasm-interface-evidence",
  "reproducible-build-evidence",
];

describe("browser Clang-WASM build-input lock", () => {
  it("strict-decodes one pinned input-only authority with stable hashes", async () => {
    const resource = cppCuteBrowserBuildInputLockResourceBytes();
    const prepared = await decodeCppCuteBrowserBuildInputLock(resource);

    expect(prepared).toEqual({
      lockId: LOCK_ID,
      resourceSha256: RESOURCE_SHA256,
      recipeSha256: RECIPE_SHA256,
      extractorSourceSetSha256: EXTRACTOR_SOURCE_SHA256,
      noticeInventorySha256: NOTICE_SHA256,
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
      runtimeAbiResourceSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      runtimeAbiResourceByteLength: cppCuteBrowserRuntimeAbiManifestResourceBytes().byteLength,
      resourceByteLength: 22_403,
      releaseReady: false,
      releaseBlockerIds: BLOCKERS,
    });
    expect(canonicalCppCuteBrowserBuildInputLockBytes(prepared)).toEqual(resource);
    expect(await deriveCppCuteBrowserBuildInputLockId(
      unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body,
    )).toBe(LOCK_ID);
  });

  it("binds exact canonical ABI bytes into one deterministic distribution output", async () => {
    const build = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const body = unwrapPreparedCppCuteBrowserBuildInputLock(build).lock.body;

    expect(body.runtimeAbiResource).toEqual({
      outputPath: "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      runtimeAbiId: runtimeAbi.runtimeAbiId,
      manifestId: runtimeAbi.manifestId,
      resourceSha256: runtimeAbi.resourceSha256,
      resourceByteLength: String(runtimeAbi.resourceByteLength),
      byteIdentity: "must-equal-package-canonical-resource",
      authority: "design-reference-only-no-wasm-conformance-worker-or-release-authority",
    });
    expect(build).toMatchObject({
      runtimeAbiManifestId: runtimeAbi.manifestId,
      runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
      runtimeAbiResourceByteLength: runtimeAbi.resourceByteLength,
      releaseReady: false,
    });
    expect(body.releasePolicy.blockerIds).not.toContain("runtime-abi-manifest");
  });

  it("pins exact upstream archives, Git identities, and build-only OCI image", async () => {
    const prepared = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const { body } = unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock;

    expect(body.sources).toEqual([
      expect.objectContaining({
        sourceId: "cutlass",
        tag: "v3.7.0",
        commit: "b78588d1630aa6643bf021613717bafb705df4ef",
        treeSha1: "4f4ae808cf284b8b0599d520a46e9fe364b87fa6",
        archiveSha256: "dfcafb7435a1b114ce32faee4f3257e276caf08f55fea04fa8bf3efa3a83c814",
        archiveByteLength: "29728321",
      }),
      expect.objectContaining({
        sourceId: "llvm-project",
        tag: "llvmorg-22.1.8",
        commit: "ca7933e47d3a3451d81e72ac174dcb5aa28b59d1",
        treeSha1: "1e4fdb95266974a0cbca9ec4c6f740488322f238",
        archiveSha256: "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888",
        archiveByteLength: "167061596",
        attestationSha256: "dd4aa06bd73706743090631300c02a6d8a3df43d41d85c627ec438d5a13b3739",
        attestationByteLength: "11234",
      }),
    ]);
    expect(body.builder).toMatchObject({
      platform: "linux/amd64",
      platformManifestDigest: "sha256:2a7a41cd7e2065b30ba389c8db0fbeaebd7ec06bb4e20f23cab8ba92180f25c7",
      imageConfigDigest: "sha256:1998ba0793f0e61685f08c62a3e78bbcd1ef84895fefe994bf48d8d66dc1e495",
      emsdk: {
        commit: "db04e88298d9916fc51fcd3743045ca3eb695127",
        releaseBundleCommit: "9074aa513b501925adb1361e208932ad32a29a5f",
        releaseBundleSha256: "3f32b91a3f8d405846ccacee911f9364da75f413fbd11ea1f3f7f23bf9d07cf3",
        releaseBundleByteLength: "292035244",
        emscriptenCommit: "283e2d130132859fde6a4e4c87fd254b38127651",
        llvmToolchainCommit: "592953beff733a1e28f6c6e5e39f948fb035a329",
        binaryenCommit: "1517ea948c09455502ba45ee3f26ea06fb2b7542",
      },
    });
    expect(body.scope).toMatchObject({
      authority: "build-input-selection-only",
      outputIdentity: "not-authorized",
      dockerUse: "pinned-build-time-only",
      runtimeDocker: "forbidden",
      networkDuringBuild: "forbidden",
    });
  });

  it("binds native TableGen then Emscripten cross-build without runtime Docker", async () => {
    const prepared = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const recipe = unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body.recipe;

    expect(recipe.stages.map((stage) => [stage.ordinal, stage.stageId, stage.executionPlatform])).toEqual([
      [0, "native-tablegen", "linux/amd64"],
      [1, "clang-extractor-wasm", "wasm32-unknown-emscripten"],
    ]);
    expect(recipe.stages[0]?.targets).toEqual(["clang-tblgen", "llvm-tblgen"]);
    expect(recipe.stages[1]?.targets).toEqual(["browsergrad-cpp-cute-extractor"]);
    expect(recipe.parallelJobs).toBe(1);
    expect(recipe.stages[1]?.definitions).toContainEqual({
      name: "BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH",
      value: "@BUILD_EVIDENCE@/generated/clang-extractor.mjs",
    });
    expect(recipe.stages[1]?.definitions).toContainEqual({
      name: "LLVM_NATIVE_TOOL_DIR",
      value: "@NATIVE_BUILD@/bin",
    });
    expect(recipe.stages[1]?.definitions).toContainEqual({
      name: "CMAKE_TRY_COMPILE_TARGET_TYPE",
      value: "STATIC_LIBRARY",
    });
    expect(recipe.stages[1]?.linkerFlags).toContain("-sENVIRONMENT=worker");
    expect(recipe.stages[1]?.linkerFlags).toContain("-sMODULARIZE=1");
    expect(recipe.stages[1]?.linkerFlags).toContain("-sEXPORT_ES6=1");
    expect(recipe.stages[1]?.linkerFlags).not.toContain("-sSTANDALONE_WASM=1");
    expect(recipe.stages[1]?.linkerFlags).toContain("-sFILESYSTEM=0");
    expect(recipe.stages[1]?.linkerFlags).toContain("-sMAXIMUM_MEMORY=1073741824");
    expect(recipe.stages[1]?.linkerFlags).toContain("-sABORTING_MALLOC=0");
    expect(recipe.stages[1]?.linkerFlags).toContain(
      "-sEXPORTED_FUNCTIONS=['_bg_cpp_cute_abi_version','_bg_cpp_cute_alloc','_bg_cpp_cute_allocator_metrics_pointer','_bg_cpp_cute_compile','_bg_cpp_cute_free','_bg_cpp_cute_reset','_bg_cpp_cute_result_length','_bg_cpp_cute_result_pointer','_bg_cpp_cute_status']",
    );
    expect(recipe.stages[1]?.linkerFlags).toContain(
      "-Wl,--Map=@BUILD_EVIDENCE@/clang-extractor.link.map",
    );
    expect(recipe.reproducibility).toEqual({
      cleanBuildCount: 2,
      sourceAndBuildPaths: "distinct",
      comparison: "per-output-reproducibility-class",
      deterministicSubjects: "sha256-byte-for-byte-across-clean-builds",
      detachedEvidence: "envelope-bytes-may-differ-subject-sha256-must-match",
      nativeTablegenIdentity: "record-and-compare-sha256",
      buildLogs: "canonical-command-and-environment-records-required",
      attestation: "externally-signed-detached-required",
    });
    expect(unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body.unresolvedBuildInputs)
      .toContainEqual({
        blockerId: "browsergrad-extractor-runtime-metrics-export",
        requirement:
          "runtime-abi-1.1-metrics-source-exists-but-pinned-executed-wasm-interception-zero-size-reallocation-and-call-graph-conformance-remain-unproved",
      });
  });

  it("closes ambient discovery, extractor links, and every distributed output path", async () => {
    const prepared = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const { recipe, notices } = unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body;

    for (const stage of recipe.stages) {
      expect(stage.definitions).toContainEqual({
        name: "CMAKE_FIND_USE_PACKAGE_REGISTRY",
        value: "OFF",
      });
      expect(stage.definitions).toContainEqual({
        name: "CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH",
        value: "OFF",
      });
      expect(stage.definitions).toContainEqual({
        name: "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY",
        value: "OFF",
      });
      expect(stage.definitions).toContainEqual({
        name: "CMAKE_MAKE_PROGRAM",
        value: "/usr/bin/make",
      });
      expect(stage.definitions).toContainEqual({
        name: "Python3_EXECUTABLE",
        value: "/usr/bin/python3",
      });
    }

    expect(recipe.extractorLinkPolicy).toEqual({
      selectedClangLibraries: [
        "clangAST",
        "clangBasic",
        "clangDriver",
        "clangFrontend",
        "clangIndex",
        "clangLex",
        "clangParse",
        "clangSema",
        "clangSerialization",
        "clangTooling",
      ],
      clangDriverUse: "in-process-tooling-dependency-only",
      driverSubprocesses: "forbidden",
      transitiveDependencies: "only-cmake-declared-static-dependencies-at-pinned-llvm-source",
      linkMapObjectClosure: "detached-evidence-required",
      allocatorInterceptionPolicy: {
        exactEntrypoints: [
          "aligned_alloc", "calloc", "free", "__libc_calloc", "__libc_free",
          "__libc_malloc", "__libc_realloc", "malloc", "memalign", "posix_memalign",
          "pvalloc", "realloc", "reallocarray", "valloc",
        ],
        forbiddenEntrypoints: [
          "bulk_free", "independent_calloc", "independent_comalloc", "realloc_in_place",
        ],
        directBypassReferences:
          "forbidden-outside-BrowserGradCppCuteMetrics.cpp",
        observedCallGraph: "detached-evidence-required",
      },
      prohibitedComponents: [
        "clangCodeGen",
        "clangInterpreter",
        "CUDA driver/runtime",
        "LLD",
        "LLVMExecutionEngine",
        "LLVMJITLink",
        "LLVMNVPTXCodeGen",
        "LLVMWebAssemblyCodeGen",
        "PTX assembler",
        "user-produced WASM",
      ],
    });

    expect(recipe.distributedOutputPlan.closure).toBe(
      "exact-path-set-no-additional-distributed-files",
    );
    const outputPaths = recipe.distributedOutputPlan.outputs.map((output) => output.path);
    expect(outputPaths).toEqual([
      "assets/browsergrad-cpp-cute/asset-manifest.json",
      "assets/browsergrad-cpp-cute/build-input-lock.json",
      "assets/browsergrad-cpp-cute/build-provenance.dsse.json",
      "assets/browsergrad-cpp-cute/clang-extractor.wasm",
      "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
      "assets/browsergrad-cpp-cute/cpp-cute-browser-worker.mjs",
      "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
      "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
      "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
      "assets/browsergrad-cpp-cute/license-inventory.json",
      "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
      "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
      "assets/browsergrad-cpp-cute/semantic-adapter-manifest.json",
      "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt",
      "licenses/browsergrad-compiler.LICENSE",
      "licenses/clang.LICENSE.txt",
      "licenses/compiler-rt.LICENSE.txt",
      "licenses/cutlass.LICENSE.txt",
      "licenses/emscripten.LICENSE",
      "licenses/emscripten-musl.COPYRIGHT",
      "licenses/libcxx.LICENSE.txt",
      "licenses/libcxxabi.LICENSE.txt",
      "licenses/libunwind.LICENSE.txt",
      "licenses/llvm.LICENSE.txt",
    ]);
    expect(outputPaths).not.toContain("clang-extractor.link.map");
    expect(outputPaths.some((path) => path.endsWith("/clang-extractor.link.map"))).toBe(false);
    expect(notices.approvedComponents.map((entry) => entry.noticeOutputPath)).toEqual(
      outputPaths.filter((path) => path.startsWith("licenses/")),
    );
    expect(recipe.distributedOutputPlan.outputs.find(
      (output) => output.role === "detached-build-provenance",
    )).toEqual({
      path: "assets/browsergrad-cpp-cute/build-provenance.dsse.json",
      role: "detached-build-provenance",
      mediaType: "application/vnd.dsse.envelope.v1+json",
      reproducibilityClass: "detached-evidence",
    });
    expect(recipe.distributedOutputPlan.outputs.every(
      (output) => output.role === "detached-build-provenance" ||
        output.reproducibilityClass === "deterministic-subject",
    )).toBe(true);
    expect(recipe.distributedOutputPlan.outputs.find(
      (output) => output.role === "runtime-abi-manifest",
    )).toMatchObject({
      path: "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      reproducibilityClass: "deterministic-subject",
    });
  });

  it("pins reviewed notices but leaves CUDA, sysroot, and file-level closure unresolved", async () => {
    const prepared = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const notices = unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body.notices;

    expect(notices.approvedComponents.map((entry) => entry.componentId)).toEqual([
      "browsergrad-compiler",
      "clang",
      "compiler-rt",
      "cutlass",
      "emscripten",
      "emscripten-musl",
      "libcxx",
      "libcxxabi",
      "libunwind",
      "llvm",
    ]);
    expect(notices.approvedComponents.find((entry) => entry.componentId === "clang")).toMatchObject({
      licenseExpression: "Apache-2.0 WITH LLVM-exception",
      noticeSha256: "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47",
      noticeByteLength: "15140",
    });
    expect(notices.unresolvedComponents).toEqual([
      {
        componentId: "cuda-toolkit-12.6.3-headers",
        intendedAsset: "dependency-header-pack:cuda",
        reasonCode: "exact-file-redistribution-review-required",
        disposition: "blocks-release",
      },
      {
        componentId: "linux-sysroot",
        intendedAsset: "dependency-header-pack:linux-sysroot",
        reasonCode: "source-package-license-and-redistribution-closure-required",
        disposition: "blocks-release",
      },
    ]);
    expect(notices.fileInventoryPolicy).toBe("exact-distributed-file-to-notice-map-required");
  });

  it("cannot self-promote input selection into build provenance or release authority", async () => {
    const prepared = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const policy = unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body.releasePolicy;

    expect(cppCuteBrowserBuildReleaseReadiness(prepared)).toEqual({
      ready: false,
      blockerIds: BLOCKERS,
    });
    expect(policy.requiredExternalAuthorities).toEqual(["canonical-runtime-abi-manifest"]);
    expect(policy.requiredDetachedEvidence).toContain(
      "observed-wasm-import-export-feature-and-memory-projection",
    );
    expect(() => assertCppCuteBrowserBuildReleaseReady(prepared)).toThrow(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RELEASE-BLOCKED",
      path: "$.body.releasePolicy",
    }));
  });

  it.each([
    ["LLVM archive hash", (input: MutableResource) => {
      const sources = body(input).sources as Record<string, unknown>[];
      const llvm = sources.find((entry) => entry.sourceId === "llvm-project");
      if (llvm !== undefined) llvm.archiveSha256 = "0".repeat(64);
    }],
    ["emsdk platform digest", (input: MutableResource) => {
      objectField(body(input), "builder").platformManifestDigest = `sha256:${"0".repeat(64)}`;
    }],
    ["cross-build target", (input: MutableResource) => {
      const recipe = objectField(body(input), "recipe");
      const stages = recipe.stages as Record<string, unknown>[];
      const wasm = stages[1];
      if (wasm !== undefined) wasm.targets = ["clang"];
    }],
    ["notice digest", (input: MutableResource) => {
      const notices = objectField(body(input), "notices");
      const approved = notices.approvedComponents as Record<string, unknown>[];
      const clang = approved.find((entry) => entry.componentId === "clang");
      if (clang !== undefined) clang.noticeSha256 = "0".repeat(64);
    }],
    ["CUDA disposition", (input: MutableResource) => {
      const notices = objectField(body(input), "notices");
      const unresolved = notices.unresolvedComponents as Record<string, unknown>[];
      const cuda = unresolved.find((entry) => entry.componentId === "cuda-toolkit-12.6.3-headers");
      if (cuda !== undefined) cuda.disposition = "approved";
    }],
    ["release decision", (input: MutableResource) => {
      objectField(body(input), "releasePolicy").decision = "ready";
    }],
    ["runtime Docker", (input: MutableResource) => {
      objectField(body(input), "scope").runtimeDocker = "allowed";
    }],
    ["runtime ABI resource hash", (input: MutableResource) => {
      objectField(body(input), "runtimeAbiResource").resourceSha256 = "0".repeat(64);
    }],
    ["runtime ABI release authority", (input: MutableResource) => {
      objectField(body(input), "runtimeAbiResource").authority = "release-ready";
    }],
  ])("rejects mutated supported selection: %s", async (_name, mutate) => {
    const input = mutableResource();
    mutate(input);
    await expect(decodeCppCuteBrowserBuildInputLock(canonicalJsonBytes(input))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
    });
  });

  it.each([
    ["duplicate definition", (input: MutableResource) => {
      const stages = objectField(body(input), "recipe").stages as Record<string, unknown>[];
      const definitions = stages[0]?.definitions as Record<string, unknown>[];
      definitions.push({ ...definitions[0] });
    }, "$.body.recipe.stages[0].definitions[*].name"],
    ["unsafe output path", (input: MutableResource) => {
      const outputs = objectField(objectField(body(input), "recipe"), "distributedOutputPlan")
        .outputs as Record<string, unknown>[];
      if (outputs[0] !== undefined) outputs[0].path = "../escape";
    }, "$.body.recipe.distributedOutputPlan.outputs[*].path"],
    ["noncanonical WireU64", (input: MutableResource) => {
      const sources = body(input).sources as Record<string, unknown>[];
      if (sources[0] !== undefined) sources[0].archiveByteLength = "01";
    }, "$.body.sources[0].archiveByteLength"],
    ["unknown recipe placeholder", (input: MutableResource) => {
      const stages = objectField(body(input), "recipe").stages as Record<string, unknown>[];
      const definitions = stages[0]?.definitions as Record<string, unknown>[];
      if (definitions[0] !== undefined) definitions[0].value = "@HOME@";
    }, "$.body.recipe.stages[0].definitions[0].value"],
    ["notice outside license outputs", (input: MutableResource) => {
      const approved = objectField(body(input), "notices").approvedComponents as Record<string, unknown>[];
      if (approved[0] !== undefined) {
        approved[0].noticeOutputPath = "assets/browsergrad-cpp-cute/asset-manifest.json";
      }
    }, "$.body.notices.approvedComponents[0].noticeOutputPath"],
  ])("rejects violated intrinsic invariant: %s", async (_name, mutate, path) => {
    const input = mutableResource();
    mutate(input);
    await expect(decodeCppCuteBrowserBuildInputLock(canonicalJsonBytes(input))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
      path,
    });
  });

  it("rejects noncanonical bytes, unknown fields, versions, and wrong IDs", async () => {
    const canonical = cppCuteBrowserBuildInputLockResourceBytes();
    const noncanonical = new Uint8Array(canonical.byteLength + 1);
    noncanonical.set(canonical);
    noncanonical[canonical.byteLength] = 0x0a;
    await expect(decodeCppCuteBrowserBuildInputLock(noncanonical)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-NONCANONICAL-BYTES",
      path: "$bytes",
    });

    const unknown = mutableResource();
    unknown.unknown = true;
    await expect(decodeCppCuteBrowserBuildInputLock(canonicalJsonBytes(unknown))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
      path: "$",
    });

    const version = mutableResource();
    objectField(version, "version").major = 2;
    await expect(decodeCppCuteBrowserBuildInputLock(canonicalJsonBytes(version))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNSUPPORTED-VERSION",
      path: "$.version.major",
    });

    const id = mutableResource();
    id.lockId = `bg.cpp.browser-build-input-lock.sha256.${"0".repeat(64)}`;
    await expect(decodeCppCuteBrowserBuildInputLock(canonicalJsonBytes(id))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-HASH-MISMATCH",
      path: "$.lockId",
    });
  });

  it("bounds bytes, rejects shared/subclass inputs, and snapshots resource copies", async () => {
    await expect(decodeCppCuteBrowserBuildInputLock(
      new Uint8Array(CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT + 1),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RESOURCE-LIMIT",
      path: "$bytes",
    });

    class ByteSubclass extends Uint8Array {}
    await expect(decodeCppCuteBrowserBuildInputLock(new ByteSubclass(16))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
      path: "$bytes",
    });

    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(decodeCppCuteBrowserBuildInputLock(
        new Uint8Array(new SharedArrayBuffer(16)),
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
        path: "$bytes",
      });
    }

    const first = cppCuteBrowserBuildInputLockResourceBytes();
    first.fill(0);
    const second = cppCuteBrowserBuildInputLockResourceBytes();
    expect(second[0]).toBe(0x7b);
    await expect(decodeCppCuteBrowserBuildInputLock(second)).resolves.toMatchObject({ lockId: LOCK_ID });
  });

  it("rejects cancelled, hostile options and forged authorities", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-CANCELLED",
      path: "$options.signal",
    });

    const hostile = Object.create(null) as Record<string, unknown>;
    Object.defineProperty(hostile, "signal", {
      enumerable: true,
      get: () => controller.signal,
    });
    await expect(decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
      hostile,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID",
      path: "$options",
    });

    const forged = {
      lockId: LOCK_ID,
      releaseReady: false,
      releaseBlockerIds: BLOCKERS,
    } as unknown as PreparedCppCuteBrowserBuildInputLock;
    expect(() => unwrapPreparedCppCuteBrowserBuildInputLock(forged)).toThrow(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNVERIFIED",
      path: "$prepared",
    }));
  });
});

type MutableResource = Record<string, unknown>;

function mutableResource(): MutableResource {
  return structuredClone(decodeWireJson(
    cppCuteBrowserBuildInputLockResourceBytes(),
  )) as unknown as MutableResource;
}

function body(input: MutableResource): MutableResource {
  return objectField(input, "body");
}

function objectField(input: MutableResource, key: string): MutableResource {
  const value = input[key];
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${key} is not an object`);
  }
  return value as MutableResource;
}
