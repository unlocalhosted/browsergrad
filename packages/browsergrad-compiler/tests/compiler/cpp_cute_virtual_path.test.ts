import { describe, expect, it } from "vitest";
import {
  prepareCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  prepareCppCuteFrontendRequest,
} from "../../src/cpp_cute_frontend_request.js";
import {
  CPP_CUTE_VIRTUAL_PATH_MAXIMUM_UTF8_BYTE_LENGTH,
  findCppCuteVirtualPathError,
} from "../../src/cpp_cute_virtual_path.js";
import { verifyCppCuteFrontendArtifact } from "../../src/cpp_cute_frontend_artifact.js";
import {
  cloneCppCuteArtifactInput,
  cloneCppCuteProfileInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

const HOSTILE_PATHS = [
  "/workspace/src/control-\u0001.cu",
  "/workspace/src/control-\u001f.cu",
  "/workspace/src/del-\u007f.cu",
  `/${"é".repeat(2_048)}`,
] as const;

describe("C++/CuTe canonical virtual paths", () => {
  it("uses UTF-8 bytes and rejects every native-VFS-forbidden path class", () => {
    const exactLimit = `/${"é".repeat(2_047)}a`;
    expect(new TextEncoder().encode(exactLimit)).toHaveLength(
      CPP_CUTE_VIRTUAL_PATH_MAXIMUM_UTF8_BYTE_LENGTH,
    );
    expect(findCppCuteVirtualPathError(exactLimit)).toBeNull();
    expect(findCppCuteVirtualPathError(`/${"é".repeat(2_048)}`)).toContain("4096 UTF-8 bytes");

    for (const path of [
      "",
      "relative/path.cu",
      "/workspace\\main.cu",
      "/workspace/\u0000/main.cu",
      "/workspace/\u0001/main.cu",
      "/workspace/\u001f/main.cu",
      "/workspace/\u007f/main.cu",
      "/workspace//main.cu",
      "/workspace/./main.cu",
      "/workspace/../main.cu",
      "/workspace/main.cu/",
      "/workspace/\ud800/main.cu",
    ]) {
      expect(findCppCuteVirtualPathError(path), path).not.toBeNull();
    }
    expect(findCppCuteVirtualPathError("/")).toBeNull();
    expect(findCppCuteVirtualPathError("/workspace/😀/main.cu")).toBeNull();
  });

  it.each(HOSTILE_PATHS)("rejects %j identically across profile, request, and artifact boundaries", async (path) => {
    const reason = findCppCuteVirtualPathError(path);
    if (reason === null) throw new Error("hostile fixture became valid");
    const profileInput = cloneCppCuteProfileInput();
    (profileInput["virtualFileSystem"] as Record<string, unknown>)["sourceRoots"] = [path];
    await expect(prepareCppCuteFrontendProfile(profileInput)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.virtualFileSystem.sourceRoots[0]",
      message: expect.stringContaining(reason),
    });

    const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
    await expect(prepareCppCuteFrontendRequest(profile, {
      schema: "browsergrad.compiler.cpp-cute.frontend-request",
      version: { major: 1, minor: 0 },
      requestId: `bg.cpp.frontend-request.sha256.${"0".repeat(64)}`,
      compilationContractHash: profile.compilationContractHash,
      mainVirtualPath: path,
      files: [],
      entryRequests: [],
      expectedArtifact: {},
      limits: {},
    }, [])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      path: "$.mainVirtualPath",
      message: expect.stringContaining(reason),
    });

    const artifact = await cloneCppCuteArtifactInput();
    const inputs = (artifact["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    const includeRoots = inputs["includeRoots"] as Record<string, unknown>[];
    if (includeRoots[0] === undefined) throw new Error("fixture lost include root");
    includeRoots[0]["virtualPath"] = path;
    await expect(verifyCppCuteFrontendArtifact(artifact)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.inputs.includeRoots[0].virtualPath",
      message: expect.stringContaining(reason),
    });
  });
});
