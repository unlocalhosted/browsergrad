import { createWgslFloat16Array } from "../packages/browsergrad-kernels/dist/index.js";

export function syntheticInputForCompiled(compiled) {
  const scalars = {};
  const buffers = {};
  const constants = {};
  const deviceGlobals = {};
  const memoryPools = {};
  const textures = {};
  const surfaces = {};
  for (const param of compiled.ir.params) {
    if (param.valueType === "surface2d") {
      surfaces[param.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
    } else if (param.valueType === "texture2d") {
      textures[param.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
    } else if (param.pointer) {
      if (param.valueType === "devicepool") {
        memoryPools[param.name] = { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
      } else {
        buffers[param.name] = syntheticBufferForType(param.valueType, 4096, compiled.f16Mode);
      }
    } else {
      scalars[param.name] = syntheticScalarForName(param.name);
    }
  }
  for (const constant of compiled.ir.constants) {
    if (constant.init !== undefined) continue;
    constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
      ? syntheticScalarForName(constant.name)
      : syntheticConstantBufferForType(constant.valueType, constant.name, 4096, compiled.f16Mode);
  }
  for (const global of compiled.ir.deviceGlobals) {
    const length = global.dimensions.length === 0
      ? 1
      : global.dimensions.reduce((product, dimension) => product * dimension, 1);
    deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length, compiled.f16Mode);
  }
  for (const texture of compiled.ir.textures) {
    textures[texture.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
  }
  for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
    memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
  }
  return { buffers, scalars, constants, deviceGlobals, memoryPools, textures, surfaces };
}

export function syntheticLaunchForCompiled(compiled) {
  return { gridDim: [1, 1, 1], blockDim: compiled.ir.workgroupSize };
}

export function externalDevicePoolNamesFromSource(source) {
  return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
    .map((match) => match[1])
    .filter(Boolean);
}

export function syntheticBufferForType(type, length = 4096, f16Mode = "native") {
  if (type === "int" || /^int[234]$/u.test(type)) return new Int32Array(length);
  if (type === "uint" || type === "uchar" || /^uint[234]$/u.test(type) || type === "voidptr" || type === "bool") return new Uint32Array(length);
  if ((type === "half" || type === "half2") && f16Mode !== "f32") return createWgslFloat16Array(length);
  return new Float32Array(length);
}

export function syntheticConstantBufferForType(type, name, length = 4096, f16Mode = "native") {
  const buffer = syntheticBufferForType(type, length, f16Mode);
  if (buffer instanceof Uint32Array || buffer instanceof Int32Array) {
    buffer.fill(/gridSize|numCells|numBodies|maxParticlesPerCell/iu.test(name) ? 64 : 1);
  } else if (buffer instanceof Float32Array) {
    buffer.fill(/worldOrigin|colliderPos|gravity/iu.test(name) ? 0 : 1);
  }
  return buffer;
}

export function isCudaVectorTypeName(type) {
  return /^(?:float|int|uint)[234]$|^half2$|^bf162$/u.test(type);
}

export function syntheticScalarForName(name) {
  if (/^(?:warpSize|warp_size)$/u.test(name)) return 32;
  if (/^(?:nanoseconds|microseconds|milliseconds)$/iu.test(name)) return 0;
  if (/(?:clock|delay|sleep|spin|wait)/iu.test(name)) return 0;
  if (/^(?:depth|level)$/iu.test(name)) return 0;
  if (/^(?:maxDepth|max_depth|maxLevel|max_level)$/u.test(name)) return 4;
  if (/^(?:left|begin|start|offset)$/u.test(name)) return 0;
  if (/^(?:right|end|len|nLines|nTessPoints)$/u.test(name)) return 64;
  if (/^(?:C|cols|columns|channels|nChannels|vocabSize|vocab_size)$/u.test(name)) return 64;
  if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize|size)$/u.test(name)) return 1024;
  if (/^(?:threads|threadsPerBlock|threads_per_block|blockSize|block_size)$/u.test(name)) return 256;
  if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
  return 1;
}
