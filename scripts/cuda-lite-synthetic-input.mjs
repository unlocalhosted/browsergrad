import { createWgslFloat16Array } from "../packages/browsergrad-kernels/dist/index.js";

export function syntheticInputForCompiled(compiled) {
  const scalars = {};
  const buffers = {};
  const constants = {};
  const deviceGlobals = {};
  const memoryPools = {};
  const textures = {};
  const surfaces = {};
  for (const param of compiled.kernelIr.params) {
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
  for (const constant of compiled.ast.constants) {
    if (constant.init !== undefined) continue;
    constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
      ? syntheticScalarForName(constant.name)
      : syntheticConstantBufferForType(constant.valueType, constant.name, 4096, compiled.f16Mode);
  }
  for (const global of compiled.ast.deviceGlobals) {
    const length = global.dimensions.length === 0
      ? 1
      : global.dimensions.reduce((product, dimension) => product * dimension, 1);
    deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length, compiled.f16Mode);
  }
  for (const texture of compiled.ast.textures) {
    textures[texture.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
  }
  for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
    memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
  }
  return { buffers, scalars, constants, deviceGlobals, memoryPools, textures, surfaces };
}

export function syntheticLaunchForCompiled(compiled) {
  return { gridDim: [1, 1, 1], blockDim: compiled.kernelIr.workgroupSize };
}

export function externalDevicePoolNamesFromSource(source) {
  return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
    .map((match) => match[1])
    .filter(Boolean);
}

export function syntheticBufferForType(type, length = 4096, f16Mode = "native") {
  if (type === "int" || /^int[234]$/u.test(type)) return seedSyntheticBuffer(new Int32Array(length), type);
  if (type === "uint" || type === "uchar" || /^uint[234]$/u.test(type) || type === "bool") {
    return seedSyntheticBuffer(new Uint32Array(length), type);
  }
  if (type === "voidptr") return new Uint32Array(length);
  if ((type === "half" || type === "half2") && f16Mode !== "f32") {
    return seedSyntheticBuffer(createWgslFloat16Array(length), type);
  }
  return seedSyntheticBuffer(new Float32Array(length), type);
}

export function seedSyntheticBuffer(buffer, type) {
  for (let index = 0; index < buffer.length; index += 1) {
    if (buffer instanceof Uint32Array) {
      buffer[index] = type === "bool" ? index % 2 : (index % 8) + 1;
    } else if (buffer instanceof Int32Array) {
      buffer[index] = (index % 8) + 1;
    } else {
      buffer[index] = ((index % 8) + 1) * 0.25;
    }
  }
  return buffer;
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
  if (/^(?:T|seqLen|seq_len|timeSteps|time_steps)$/u.test(name)) return 4;
  if (/^(?:B|batch|batchSize|batch_size)$/u.test(name)) return 1;
  if (/^(?:NH|numHeads|num_heads|heads)$/u.test(name)) return 1;
  if (/^(?:HS|headSize|head_size)$/u.test(name)) return 4;
  if (/^(?:C|cols|columns|channels|nChannels|vocabSize|vocab_size)$/u.test(name)) return 4;
  if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize|size)$/u.test(name)) return 1024;
  if (/^(?:threads|threadsPerBlock|threads_per_block|blockSize|block_size)$/u.test(name)) return 256;
  if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
  return 1;
}
