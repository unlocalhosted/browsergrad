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

add_llvm_executable(browsergrad-cpp-cute-extractor
  BrowserGradCppCuteArtifactV3.cpp
  BrowserGradCppCuteArtifactV3.h
  BrowserGradCppCuteClangAction.cpp
  BrowserGradCppCuteClangAction.h
  BrowserGradCppCuteExtractor.cpp
  BrowserGradCppCuteImportedVfs.cpp
  BrowserGradCppCuteImportedVfs.h
  BrowserGradCppCuteMetrics.cpp
  BrowserGradCppCuteMetrics.h
  BrowserGradCppCuteRuntime.cpp
  BrowserGradCppCuteRuntime.h
)

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
