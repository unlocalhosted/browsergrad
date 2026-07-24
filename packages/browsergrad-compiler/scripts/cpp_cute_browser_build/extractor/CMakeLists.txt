cmake_minimum_required(VERSION 3.20.0)

project(browsergrad_extractor LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "browsergrad_extractor must be cross-compiled with the pinned Emscripten toolchain")
endif()

if(NOT DEFINED BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH)
  message(FATAL_ERROR "BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH must name the locked Emscripten factory module")
endif()

get_filename_component(BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_DIRECTORY
  "${BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH}" DIRECTORY)
get_filename_component(BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_FILENAME
  "${BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH}" NAME)
if(NOT BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_FILENAME STREQUAL "clang-extractor.mjs")
  message(FATAL_ERROR "the locked extractor factory filename must be clang-extractor.mjs")
endif()

# LLVM and Clang remain exception-free. BrowserGrad's boundary converts its own
# allocation and validation exceptions into the closed C ABI status model, so
# this one target requires Emscripten's JavaScript exception implementation.
# The locked compiler and linker flags select that implementation explicitly;
# native Wasm exception tags remain forbidden by runtime ABI v1.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

# Clang 22 exposes exact preprocessing and template callbacks but not the
# consumed constexpr-step count. Build one hash-verified replacement for the
# ExprConstant.cpp archive member and inject a single fail-closed hook at its
# existing step boundary. The upstream source archive remains immutable.
set(BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SOURCE
  "${LLVM_MAIN_SRC_DIR}/../clang/lib/AST/ExprConstant.cpp")
if(NOT EXISTS "${BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SOURCE}")
  message(FATAL_ERROR "the pinned Clang ExprConstant.cpp source is missing")
endif()
file(SHA256 "${BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SOURCE}"
  BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SHA256)
if(NOT BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SHA256 STREQUAL
    "50c474805f9914a2c54d75f2ddd56d82dc20c128069855155f3c37db26e5bcf9")
  message(FATAL_ERROR "the pinned Clang ExprConstant.cpp identity changed")
endif()
file(READ "${BROWSERGRAD_UPSTREAM_EXPR_CONSTANT_SOURCE}"
  BROWSERGRAD_EXPR_CONSTANT_CONTENT)
set(BROWSERGRAD_EXPR_CONSTANT_DECLARATION_ANCHOR
  "#define DEBUG_TYPE \"exprconstant\"\n")
set(BROWSERGRAD_EXPR_CONSTANT_DECLARATION_REPLACEMENT
  "#define DEBUG_TYPE \"exprconstant\"\n\nextern \"C\" bool browsergrad_cpp_cute_constexpr_step_hook() noexcept;\n")
set(BROWSERGRAD_EXPR_CONSTANT_STEP_ANCHOR
  "    bool nextStep(const Stmt *S) {\n      if (Ctx.getLangOpts().ConstexprStepLimit == 0)\n        return true;\n")
set(BROWSERGRAD_EXPR_CONSTANT_STEP_REPLACEMENT
  "    bool nextStep(const Stmt *S) {\n      if (!browsergrad_cpp_cute_constexpr_step_hook()) {\n        FFDiag(S->getBeginLoc(), diag::note_constexpr_step_limit_exceeded);\n        return false;\n      }\n      if (Ctx.getLangOpts().ConstexprStepLimit == 0)\n        return true;\n")
function(browsergrad_require_unique_literal CONTENT_VARIABLE ANCHOR_VARIABLE)
  string(FIND "${${CONTENT_VARIABLE}}" "${${ANCHOR_VARIABLE}}"
    BROWSERGRAD_PATCH_FIRST_INDEX)
  if(BROWSERGRAD_PATCH_FIRST_INDEX EQUAL -1)
    message(FATAL_ERROR "the pinned Clang constexpr instrumentation anchor changed")
  endif()
  string(LENGTH "${${ANCHOR_VARIABLE}}" BROWSERGRAD_PATCH_ANCHOR_LENGTH)
  math(EXPR BROWSERGRAD_PATCH_REMAINDER_INDEX
    "${BROWSERGRAD_PATCH_FIRST_INDEX} + ${BROWSERGRAD_PATCH_ANCHOR_LENGTH}")
  string(SUBSTRING "${${CONTENT_VARIABLE}}"
    ${BROWSERGRAD_PATCH_REMAINDER_INDEX} -1 BROWSERGRAD_PATCH_REMAINDER)
  string(FIND "${BROWSERGRAD_PATCH_REMAINDER}" "${${ANCHOR_VARIABLE}}"
    BROWSERGRAD_PATCH_SECOND_INDEX)
  if(NOT BROWSERGRAD_PATCH_SECOND_INDEX EQUAL -1)
    message(FATAL_ERROR "the pinned Clang constexpr instrumentation anchor is not unique")
  endif()
endfunction()
browsergrad_require_unique_literal(BROWSERGRAD_EXPR_CONSTANT_CONTENT
  BROWSERGRAD_EXPR_CONSTANT_DECLARATION_ANCHOR)
browsergrad_require_unique_literal(BROWSERGRAD_EXPR_CONSTANT_CONTENT
  BROWSERGRAD_EXPR_CONSTANT_STEP_ANCHOR)
string(REPLACE "${BROWSERGRAD_EXPR_CONSTANT_DECLARATION_ANCHOR}"
  "${BROWSERGRAD_EXPR_CONSTANT_DECLARATION_REPLACEMENT}"
  BROWSERGRAD_EXPR_CONSTANT_CONTENT "${BROWSERGRAD_EXPR_CONSTANT_CONTENT}")
string(REPLACE "${BROWSERGRAD_EXPR_CONSTANT_STEP_ANCHOR}"
  "${BROWSERGRAD_EXPR_CONSTANT_STEP_REPLACEMENT}"
  BROWSERGRAD_EXPR_CONSTANT_CONTENT "${BROWSERGRAD_EXPR_CONSTANT_CONTENT}")
set(BROWSERGRAD_PATCHED_EXPR_CONSTANT_SOURCE
  "${CMAKE_CURRENT_BINARY_DIR}/BrowserGradPatchedExprConstant.cpp")
file(WRITE "${BROWSERGRAD_PATCHED_EXPR_CONSTANT_SOURCE}"
  "${BROWSERGRAD_EXPR_CONSTANT_CONTENT}")
file(SHA256 "${BROWSERGRAD_PATCHED_EXPR_CONSTANT_SOURCE}"
  BROWSERGRAD_PATCHED_EXPR_CONSTANT_SHA256)
if(NOT BROWSERGRAD_PATCHED_EXPR_CONSTANT_SHA256 STREQUAL
    "0df625b20b3ad1d72fbdfea8c72a2784f22a0387904acd3250b7810e195fa623")
  message(FATAL_ERROR "the generated constexpr instrumentation patch changed")
endif()
unset(BROWSERGRAD_EXPR_CONSTANT_CONTENT)

add_llvm_executable(browsergrad-cpp-cute-extractor
  "${BROWSERGRAD_PATCHED_EXPR_CONSTANT_SOURCE}"
  BrowserGradCppCuteArtifactV3.cpp
  BrowserGradCppCuteArtifactV3.h
  BrowserGradCppCuteArtifactJson.cpp
  BrowserGradCppCuteArtifactJson.h
  BrowserGradCppCuteArtifactWriter.cpp
  BrowserGradCppCuteArtifactWriter.h
  BrowserGradCppCuteBrowserHost.cpp
  BrowserGradCppCuteCanonicalJson.cpp
  BrowserGradCppCuteCanonicalJson.h
  BrowserGradCppCuteClangAction.cpp
  BrowserGradCppCuteClangAction.h
  BrowserGradCppCuteCommandLine.cpp
  BrowserGradCppCuteCommandLine.h
  BrowserGradCppCuteCommandLinePolicy.inc
  BrowserGradCppCuteCompilePlan.cpp
  BrowserGradCppCuteCompilePlan.h
  BrowserGradCppCuteCompileSession.cpp
  BrowserGradCppCuteCompileSession.h
  BrowserGradCppCuteDiagnostics.cpp
  BrowserGradCppCuteDiagnostics.h
  BrowserGradCppCuteDiagnosticsPolicy.inc
  BrowserGradCppCuteExtractor.cpp
  BrowserGradCppCuteImportedVfs.cpp
  BrowserGradCppCuteImportedVfs.h
  BrowserGradCppCuteInvocation.cpp
  BrowserGradCppCuteInvocation.h
  BrowserGradCppCuteMetrics.cpp
  BrowserGradCppCuteMetrics.h
  BrowserGradCppCutePreprocessorPolicy.cpp
  BrowserGradCppCutePreprocessorPolicy.h
  BrowserGradCppCuteProducer.cpp
  BrowserGradCppCuteProducer.h
  BrowserGradCppCuteProducerObservation.cpp
  BrowserGradCppCuteProducerObservation.h
  BrowserGradCppCuteRuntime.cpp
  BrowserGradCppCuteRuntime.h
  BrowserGradCppCuteSha256.cpp
  BrowserGradCppCuteSha256.h
  BrowserGradCppCuteViewCopyArtifact.cpp
  BrowserGradCppCuteViewCopyArtifact.h
  BrowserGradCppCuteVirtualPath.cpp
  BrowserGradCppCuteVirtualPath.h
)

# LLVM external projects are configured as siblings of Clang rather than below
# clang/CMakeLists.txt, so they do not inherit Clang's directory-scoped source
# and generated-header include paths. Use the exact equivalent paths owned by
# the in-tree Clang build and fail at configuration if that layout drifts.
set(BROWSERGRAD_CLANG_SOURCE_INCLUDE_CANDIDATE
  "${LLVM_MAIN_SRC_DIR}/../clang/include")
set(BROWSERGRAD_CLANG_AST_SOURCE_INCLUDE_CANDIDATE
  "${LLVM_MAIN_SRC_DIR}/../clang/lib/AST")
set(BROWSERGRAD_CLANG_BINARY_INCLUDE_CANDIDATE
  "${LLVM_BINARY_DIR}/tools/clang/include")
if(NOT IS_DIRECTORY "${BROWSERGRAD_CLANG_SOURCE_INCLUDE_CANDIDATE}")
  message(FATAL_ERROR "the pinned Clang source include directory is missing")
endif()
if(NOT IS_DIRECTORY "${BROWSERGRAD_CLANG_AST_SOURCE_INCLUDE_CANDIDATE}")
  message(FATAL_ERROR "the pinned Clang AST implementation include directory is missing")
endif()
if(NOT IS_DIRECTORY "${BROWSERGRAD_CLANG_BINARY_INCLUDE_CANDIDATE}")
  message(FATAL_ERROR "the configured Clang generated include directory is missing")
endif()
# CMake otherwise preserves the llvm/../clang spelling in flags.make. Resolve
# both existing directories before attaching them so the configured target and
# the independent pre-build reviewer bind one canonical path spelling.
file(REAL_PATH "${BROWSERGRAD_CLANG_SOURCE_INCLUDE_CANDIDATE}"
  BROWSERGRAD_CLANG_SOURCE_INCLUDE_DIRECTORY)
file(REAL_PATH "${BROWSERGRAD_CLANG_AST_SOURCE_INCLUDE_CANDIDATE}"
  BROWSERGRAD_CLANG_AST_SOURCE_INCLUDE_DIRECTORY)
file(REAL_PATH "${BROWSERGRAD_CLANG_BINARY_INCLUDE_CANDIDATE}"
  BROWSERGRAD_CLANG_BINARY_INCLUDE_DIRECTORY)
target_include_directories(browsergrad-cpp-cute-extractor PRIVATE
  "${BROWSERGRAD_CLANG_BINARY_INCLUDE_DIRECTORY}"
  "${BROWSERGRAD_CLANG_SOURCE_INCLUDE_DIRECTORY}"
  "${BROWSERGRAD_CLANG_AST_SOURCE_INCLUDE_DIRECTORY}"
)
add_dependencies(browsergrad-cpp-cute-extractor clang-tablegen-targets)

target_compile_features(browsergrad-cpp-cute-extractor PRIVATE cxx_std_20)
set_target_properties(browsergrad-cpp-cute-extractor PROPERTIES
  CXX_EXTENSIONS OFF
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME "clang-extractor"
  RUNTIME_OUTPUT_DIRECTORY "${BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_DIRECTORY}"
  SUFFIX ".mjs"
  VISIBILITY_INLINES_HIDDEN YES
)

target_link_libraries(browsergrad-cpp-cute-extractor PRIVATE
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangIndex
  clangLex
  clangParse
  clangSema
  clangSerialization
  clangTooling
)
