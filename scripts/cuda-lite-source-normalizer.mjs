import { normalizeCuteFlashAttentionKernels } from "./cuda-lite-source-normalizer-cute-flash.mjs";
import { setNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";
import * as definesNormalizers from "./cuda-lite-source-normalizer-defines.mjs";
import * as evalNormalizers from "./cuda-lite-source-normalizer-eval.mjs";
import * as inferenceNormalizers from "./cuda-lite-source-normalizer-inference.mjs";
import * as localTransformNormalizers from "./cuda-lite-source-normalizer-local-transforms.mjs";
import * as packedNormalizers from "./cuda-lite-source-normalizer-packed.mjs";
import * as podRecordNormalizers from "./cuda-lite-source-normalizer-pod-records.mjs";
import * as preprocessorNormalizers from "./cuda-lite-source-normalizer-preprocessor.mjs";
import * as recordNormalizers from "./cuda-lite-source-normalizer-records.mjs";
import * as templateSpecializationNormalizers from "./cuda-lite-source-normalizer-template-specialization.mjs";
import * as templateTransformNormalizers from "./cuda-lite-source-normalizer-template-transforms.mjs";
import * as cuteLaNormalizers from "./cuda-lite-source-normalizer-cute-la.mjs";
import { normalizeCudaCppCompat } from "./cuda-lite-source-normalizer-cpp-compat.mjs";
import {
  normalizeAtomicForwarderHelpers,
  normalizeBlockReduceHelpers,
  normalizeCurandInitOverloads,
  normalizeDeviceReferenceParams,
  normalizeLocalPointerDeviceFunctionCalls,
  normalizePointerStoreForwarderHelpers,
  normalizeSincosHelpers,
} from "./cuda-lite-source-normalizer-device-helpers.mjs";
import { normalizeDeviceRuntimeNoopCalls } from "./cuda-lite-source-normalizer-runtime.mjs";
import { normalizeWgmmaTmaGemmKernels } from "./cuda-lite-source-normalizer-wgmma.mjs";

const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "min",
  "max",
  "vec_at",
  "load128",
  "load128cs",
  "store128",
  "store128cs",
  "store128cg",
  "div_ceil",
  "blockReduce",
  "warpReduceSum",
  "warpReduceMax",
  "warpReduceMin",
  "warp_reduce_sum",
  "warp_reduce_max",
  "warp_reduce_min",
  "warp_reduce_sum_f32",
  "warp_reduce_max_f32",
  "warp_reduce_sum_f16",
  "warp_reduce_sum_f16_f16",
  "warp_reduce_sum_f16_f32",
  "warp_reduce_sum_i8_i32",
  "warp_reduce_sum_i32_i32",
  "atomicAdd",
  "atomicSub",
  "atomicMin",
  "atomicMax",
  "atomicAnd",
  "atomicOr",
  "atomicXor",
  "atomicExch",
  "atomicCAS",
]);

const SEMANTIC_TEMPLATE_CALL_STRIP_HELPERS = new Set([
  "warpReduceSum",
  "warpReduceMax",
  "warpReduceMin",
  "warp_reduce_sum",
  "warp_reduce_max",
  "warp_reduce_min",
  "warp_reduce_sum_f32",
  "warp_reduce_max_f32",
  "warp_reduce_sum_f16",
  "warp_reduce_sum_f16_f16",
  "warp_reduce_sum_f16_f32",
  "warp_reduce_sum_i8_i32",
  "warp_reduce_sum_i32_i32",
]);

const {
  collectCudaLiteContextDefines, pruneCudaPreprocessorBranches, normalizeFunctionMacro, injectSharedDeclarationsIntoKernel, kernelDefinitionName, collectKernelTemplateArguments, referencedDeviceFunctionClosure, normalizeFunctionPointerDispatch, stripFunctionPointerDeviceGlobals, sourceFunctionCallArities, sourceCallsFunctionThroughDefines, sourceCallsFunction, stripCommentsAndStrings, sourceLaunchesKernel, sourceMentionsIdentifier, reachableDeclarations, constantDeclarationName, deviceGlobalDeclarationName, textureDeclarationName, functionDefineName, recordDeclarationName, podRecordDeclarationVectorAlias, scalarizedPodRecordDeclaration, templatedScalarizedPodRecordDeclaration, bitpackedShortUnionDeclaration, kernelParamNames, isMacroIdentifier,
} = preprocessorNormalizers;
const { specializeTemplateFromLaunchContext, specializeDeviceFunctionFromCallContext, templateDefaultEnvironment, normalizeNumericObjectDefines, normalizeSupportedTypeDefineReferences } = templateSpecializationNormalizers;
const { normalizeRank2CallableViews, collectSyntheticVectorPackDefines, normalizeTemplateValueFallbacks, normalizeTemplateTypeArgument, normalizeCppTemplateCarrierSyntax, normalizeCudaVectorLength, normalizeSimpleStatementMacros, normalizeSideEffectExpressions, normalizeStdMathAliases, normalizeStaticTemplateConverters, stripTemplateRecordDeclarations, normalizeVectorCooperativeReductions, normalizeCooperativeGroupHelperParams, normalizeForLoopScopedVariables, normalizeLocalReferenceAliases } = templateTransformNormalizers;
const { normalizeCarrierMemberReferences, normalizeCudaVectorCarrierAliases, normalizeScalarizedPodRecords, normalizePodRecordVectorAliases } = recordNormalizers;
const { normalizeBitpackedShortUnions } = podRecordNormalizers;
const { normalizeLineContinuations, normalizeEscapedNewlinesOutsideStrings, normalizeSimpleLocalLambdas, normalizeSimpleExpressionMacros, normalizeDependentCarrierParams, normalizeMissingSemicolonAfterMacroAssignment, normalizeLegacyCudaArithmeticMacros, normalizeMissingThreadLocalSoftmaxNumeratorBuffers, normalizeVectorCooperativeShuffles, collectDeclaredIdentifiers, sourceUsesMemberName, normalizeSharedMemoryHelpers, addressTarget } = localTransformNormalizers;
const { normalizeCuteRank2TransposeKernels, parseCudaGlobalFunction, cudaParamName, cudaPointerParamValueType, normalizeCuteRowBroadcastGemvKernels, collectCuteIntegerEnv, normalizeCuteScalarExpression, cuteExprEquals, normalizeCuteTnGemmKernels, collectCuteExpressionAliases, resolveCuteExpressionAlias, normalizeCute1dAffineTileCopies } = cuteLaNormalizers;
const { normalizeWidePacked128Aliases, normalizeCudaPipelineAsync, normalizePacked128MemoryHelpers, normalizeVectorStaticConstructors, WIDE_PACKED128_TYPES } = packedNormalizers;
const { collectObjectDefines, collectTypeAliasDefines, collectFunctionDefineBodies, collectCarrierMemberDefines, stripSupportedTypeAliasDeclarations, stripSupportedEnumDeclarations, mergeDefineMaps } = definesNormalizers;
const { stripKernelLaunchTemplateArguments, stripKnownTemplateCallArguments, collectDeviceFunctionTemplateArguments, templateEnvironment, substituteTemplateArgument, scanTemplatedFunctionDefinitions, scanTemplatedCallReferences, collectVisiblePointerTypes, collectVisibleValueTypes, inferArgumentPointerType, inferArgumentValueType, templateParameterNames } = inferenceNormalizers;
const { evaluateTemplateIntegerExpression, splitTopLevel, balancedParenContents, findBalanced, skipWhitespace, escapeRegExp } = evalNormalizers;

const CUDA_BUILTIN_RECORD_DECLARATIONS = [
  "struct cudaExtent { size_t width; size_t height; size_t depth; };",
];

const SEMANTIC_RECORD_CARRIERS = new Set([
  "DevicePool",
]);

const normalizerHelpers = {
  ...preprocessorNormalizers, ...templateSpecializationNormalizers, ...templateTransformNormalizers, ...recordNormalizers, ...podRecordNormalizers, ...localTransformNormalizers, ...cuteLaNormalizers, ...packedNormalizers, ...definesNormalizers, ...inferenceNormalizers, ...evalNormalizers,
};
setNormalizerHelpers(normalizerHelpers);

export function createKernelCompilationUnit({
  kernel,
  siblingKernels = [],
  definesByName = new Map(),
  templateArgumentsByKernelName = new Map(),
  functionDeclarations = [],
  deviceFunctions = [],
  constantDeclarations = [],
  deviceGlobalDeclarations = [],
  textureDeclarations = [],
  sharedDeclarations = [],
  recordDeclarations = [],
  functionPointerTables = [],
}) {
  const recordContext = [
    kernel,
    ...siblingKernels,
    ...functionDeclarations,
    ...deviceFunctions.map((fn) => fn.source),
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
  ].join("\n");
  const builtinRecordDeclarations = CUDA_BUILTIN_RECORD_DECLARATIONS.filter((declaration) => {
    const name = recordDeclarationName(declaration);
    return name !== undefined && sourceMentionsIdentifier(recordContext, name);
  });
  const availableRecordDeclarations = [...builtinRecordDeclarations, ...recordDeclarations];
  const aliasContext = [
    kernel,
    ...siblingKernels,
    ...functionDeclarations,
    ...deviceFunctions.map((fn) => fn.source),
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
    ...availableRecordDeclarations,
  ].join("\n");
  const aliasDefines = collectTypeAliasDefines(aliasContext, definesByName);
  const carrierDefines = collectCarrierMemberDefines(aliasContext, mergeDefineMaps(definesByName, aliasDefines));
  const syntheticPackDefines = collectSyntheticVectorPackDefines(aliasContext, mergeDefineMaps(definesByName, aliasDefines, carrierDefines));
  const effectiveDefines = mergeDefineMaps(definesByName, aliasDefines, carrierDefines, syntheticPackDefines);
  const specializedKernel = normalizeFunctionPointerDispatch(
    specializeTemplateFromLaunchContext(kernel, templateArgumentsByKernelName, effectiveDefines),
    functionPointerTables,
  );
  const params = new Set(kernelParamNames(specializedKernel));
  const functionDefineBodies = collectFunctionDefineBodies(functionDeclarations);
  const referencedDeviceFunctionsRaw = referencedDeviceFunctionClosure(
    specializedKernel,
    deviceFunctions,
    mergeDefineMaps(effectiveDefines, functionDefineBodies),
  );
  const referencedDeviceFunctionNames = new Set(referencedDeviceFunctionsRaw.map((fn) => fn.name));
  const templateInferenceDefines = mergeDefineMaps(effectiveDefines, templateDefaultEnvironment(specializedKernel, effectiveDefines));
  const deviceTemplateArgumentsByName = collectDeviceFunctionTemplateArguments(
    [
      specializedKernel,
      ...referencedDeviceFunctionsRaw.map((fn) => fn.source),
    ],
    referencedDeviceFunctionNames,
    templateInferenceDefines,
  );
  const referencedSiblingKernelsRaw = siblingKernels.filter((sibling) => {
    const name = kernelDefinitionName(sibling);
    return name !== undefined && sourceLaunchesKernel(specializedKernel, name);
  });
  const templateNames = new Set([
    ...templateParameterNames(specializedKernel),
    ...referencedSiblingKernelsRaw.flatMap((sibling) => templateParameterNames(sibling)),
    ...referencedDeviceFunctionsRaw.flatMap((fn) => templateParameterNames(fn.source)),
  ]);
  const shadowedMacroNames = collectDeclaredIdentifiers([
    specializedKernel,
    ...referencedSiblingKernelsRaw,
    ...referencedDeviceFunctionsRaw.map((fn) => fn.source),
  ].join("\n"), effectiveDefines);
  const macroScope = [
    specializedKernel,
    ...referencedDeviceFunctionsRaw.map((fn) => fn.source),
    ...referencedSiblingKernelsRaw,
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
  ].join("\n");
  const defines = [...effectiveDefines]
    .filter(([name, value]) => isMacroIdentifier(name) && !WIDE_PACKED128_TYPES.has(value) && !params.has(name) && !templateNames.has(name) && !shadowedMacroNames.has(name) && !sourceUsesMemberName(macroScope, name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = referencedDeviceFunctionsRaw
    .map((fn) => ({
      ...fn,
      source: stripKnownTemplateCallArguments(
        stripKernelLaunchTemplateArguments(specializeDeviceFunctionFromCallContext(
          fn.source,
          deviceTemplateArgumentsByName.get(fn.name),
          templateInferenceDefines,
        )),
        referencedDeviceFunctionNames,
      ),
    }));
  const referencedSiblingKernels = referencedSiblingKernelsRaw.map((sibling) => stripKernelLaunchTemplateArguments(
    specializeTemplateFromLaunchContext(sibling, templateArgumentsByKernelName, effectiveDefines),
  ));
  const reachableDeclarationScope = [
    specializedKernel,
    ...referencedDeviceFunctions.map((fn) => fn.source),
    ...referencedSiblingKernels,
  ].join("\n");
  const referencedConstantDeclarations = reachableDeclarations(
    constantDeclarations,
    constantDeclarationName,
    reachableDeclarationScope,
  );
  const referencedDeviceGlobalDeclarations = reachableDeclarations(
    deviceGlobalDeclarations,
    deviceGlobalDeclarationName,
    reachableDeclarationScope,
  );
  const referencedTextureDeclarations = reachableDeclarations(
    textureDeclarations,
    textureDeclarationName,
    reachableDeclarationScope,
  );
  const normalizedMacroScope = [
    specializedKernel,
    ...referencedDeviceFunctions.map((fn) => fn.source),
    ...referencedSiblingKernels,
    ...referencedConstantDeclarations,
    ...referencedDeviceGlobalDeclarations,
  ].join("\n");
  const referencedRecordDeclarations = availableRecordDeclarations.filter((declaration) => {
    const name = recordDeclarationName(declaration);
    return name !== undefined &&
      sourceMentionsIdentifier(normalizedMacroScope, name) &&
      (podRecordDeclarationVectorAlias(declaration, effectiveDefines) !== undefined ||
        scalarizedPodRecordDeclaration(declaration, effectiveDefines) !== undefined ||
        templatedScalarizedPodRecordDeclaration(declaration, normalizedMacroScope) !== undefined ||
        bitpackedShortUnionDeclaration(declaration) !== undefined);
  });
  const functionMacros = functionDeclarations.filter((declaration) => {
    const name = functionDefineName(declaration);
    return name !== undefined && sourceMentionsIdentifier(normalizedMacroScope, name);
  });
  const kernelWithSharedDeclarations = injectSharedDeclarationsIntoKernel(
    stripKnownTemplateCallArguments(stripKernelLaunchTemplateArguments(specializedKernel), new Set([
      ...referencedDeviceFunctionNames,
      ...SEMANTIC_TEMPLATE_CALL_STRIP_HELPERS,
    ])),
    sharedDeclarations,
  );
  const unit = normalizeLineContinuations(normalizeEscapedNewlinesOutsideStrings([
    defines.join("\n"),
    functionMacros.map(normalizeFunctionMacro).join("\n"),
    referencedDeviceFunctions.map((fn) => fn.source).join("\n"),
    referencedConstantDeclarations.join("\n"),
    referencedDeviceGlobalDeclarations.join("\n"),
    referencedTextureDeclarations.join("\n"),
    referencedRecordDeclarations.join("\n"),
    referencedSiblingKernels.join("\n"),
    kernelWithSharedDeclarations,
  ].filter((part) => part.trim().length > 0).join("\n")));
  const prunedUnit = pruneCudaPreprocessorBranches(stripFunctionPointerDeviceGlobals(unit, functionPointerTables), effectiveDefines);
  const withCarrierMembers = normalizeCarrierMemberReferences(prunedUnit, effectiveDefines);
  const postCarrierAliasDefines = collectTypeAliasDefines(withCarrierMembers, effectiveDefines);
  const postCarrierDefines = mergeDefineMaps(effectiveDefines, postCarrierAliasDefines);
  const withVectorCarrierAliases = normalizeCudaVectorCarrierAliases(withCarrierMembers, postCarrierDefines);
  const withDeviceReferences = normalizeDeviceReferenceParams(withVectorCarrierAliases, normalizerHelpers);
  const withScalarizedRecords = normalizeScalarizedPodRecords(withDeviceReferences, postCarrierDefines);
  const withPodRecords = normalizePodRecordVectorAliases(withScalarizedRecords, postCarrierDefines);
  const withBitpackedUnions = normalizeBitpackedShortUnions(withPodRecords);
  const withNumericDefines = normalizeNumericObjectDefines(
    withBitpackedUnions,
    postCarrierDefines,
    new Set([...params, ...templateNames]),
  );
  const withCudaPipeline = normalizeCudaPipelineAsync(withNumericDefines);
  const withTypeDefines = normalizeSupportedTypeDefineReferences(
    withCudaPipeline,
    postCarrierDefines,
    new Set([...params, ...templateNames]),
  );
  const withMdspanViews = normalizeRank2CallableViews(withTypeDefines);
  const withCooperativeGroupHelpers = normalizeCooperativeGroupHelperParams(withMdspanViews);
  const withVectorCooperativeReductions = normalizeVectorCooperativeReductions(withCooperativeGroupHelpers);
  const withStdMathAliases = normalizeStdMathAliases(withVectorCooperativeReductions);
  const withStaticTemplateConverters = normalizeStaticTemplateConverters(withStdMathAliases, postCarrierDefines);
  const withoutTemplateRecords = stripTemplateRecordDeclarations(withStaticTemplateConverters);
  const withoutSupportedAliases = stripSupportedEnumDeclarations(stripSupportedTypeAliasDeclarations(withoutTemplateRecords, postCarrierDefines));
  const withVectorConstructors = normalizeVectorStaticConstructors(withoutSupportedAliases, postCarrierDefines);
  const withVectorLength = normalizeCudaVectorLength(withVectorConstructors);
  const withWidePacks = normalizeWidePacked128Aliases(withVectorLength, postCarrierDefines);
  const withPackedHelpers = normalizePacked128MemoryHelpers(withWidePacks);
  const withSharedHelpers = normalizeSharedMemoryHelpers(withPackedHelpers);
  const withCurandOverloads = normalizeCurandInitOverloads(withSharedHelpers, normalizerHelpers);
  const withRuntimeNoops = normalizeDeviceRuntimeNoopCalls(withCurandOverloads);
  const withSincosHelpers = normalizeSincosHelpers(withRuntimeNoops, normalizerHelpers);
  const withBlockReduce = normalizeBlockReduceHelpers(withSincosHelpers, normalizerHelpers);
  const withAtomicForwarders = normalizeAtomicForwarderHelpers(withBlockReduce, normalizerHelpers);
  const withPointerStoreForwarders = normalizePointerStoreForwarderHelpers(withAtomicForwarders, normalizerHelpers);
  const withLocalPointerHelpers = normalizeLocalPointerDeviceFunctionCalls(withPointerStoreForwarders, normalizerHelpers);
  const withCudaCppCompat = normalizeCudaCppCompat(withLocalPointerHelpers);
  const withLambdas = normalizeSimpleLocalLambdas(withCudaCppCompat);
  const withVectorShuffles = normalizeVectorCooperativeShuffles(withLambdas);
  const withCarrierParams = normalizeDependentCarrierParams(withVectorShuffles);
  const withWgmmaTma = normalizeWgmmaTmaGemmKernels(withCarrierParams, normalizerHelpers);
  const withCuteAttention = normalizeCuteFlashAttentionKernels(withWgmmaTma, normalizerHelpers);
  const withCuteRank2 = normalizeCuteRank2TransposeKernels(withCuteAttention);
  const withCuteGemv = normalizeCuteRowBroadcastGemvKernels(withCuteRank2);
  const withCuteGemm = normalizeCuteTnGemmKernels(withCuteGemv);
  const withCuteTiles = normalizeCute1dAffineTileCopies(withCuteGemm);
  const withExpressionMacros = normalizeSimpleExpressionMacros(withCuteTiles);
  const withRecoveredMacroStatements = normalizeMissingSemicolonAfterMacroAssignment(withExpressionMacros);
  const withLegacyArithmeticMacros = normalizeLegacyCudaArithmeticMacros(withRecoveredMacroStatements);
  const withStatementMacros = normalizeSimpleStatementMacros(withLegacyArithmeticMacros);
  const withLocalReferences = normalizeLocalReferenceAliases(withStatementMacros);
  const withSideEffects = normalizeSideEffectExpressions(withLocalReferences);
  const withScopedForVariables = normalizeForLoopScopedVariables(withSideEffects);
  const withRecoveredNumerators = normalizeMissingThreadLocalSoftmaxNumeratorBuffers(withScopedForVariables);
  const withTemplateFallbacks = normalizeTemplateValueFallbacks(withRecoveredNumerators, postCarrierDefines);
  return normalizeCppTemplateCarrierSyntax(withTemplateFallbacks);
}

export {
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
};
