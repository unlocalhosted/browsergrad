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
        buffers[param.name] = syntheticBufferForType(param.valueType);
      }
    } else {
      scalars[param.name] = syntheticScalarForName(param.name);
    }
  }
  for (const constant of compiled.ir.constants) {
    constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
      ? syntheticScalarForName(constant.name)
      : syntheticBufferForType(constant.valueType);
  }
  for (const global of compiled.ir.deviceGlobals) {
    const length = global.dimensions.length === 0
      ? 1
      : global.dimensions.reduce((product, dimension) => product * dimension, 1);
    deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length);
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

export function syntheticBufferForType(type, length = 4096) {
  if (type === "int") return new Int32Array(length);
  if (type === "uint" || type === "voidptr" || type === "bool") return new Uint32Array(length);
  return new Float32Array(length);
}

export function isCudaVectorTypeName(type) {
  return /^(?:float|int|uint)[234]$|^half2$|^bf162$/u.test(type);
}

export function syntheticScalarForName(name) {
  if (/(?:clock|delay|sleep|spin|wait)/iu.test(name)) return 0;
  if (/^(?:depth|level)$/iu.test(name)) return 0;
  if (/^(?:maxDepth|max_depth|maxLevel|max_level)$/u.test(name)) return 4;
  if (/^(?:left|begin|start|offset)$/u.test(name)) return 0;
  if (/^(?:right|end|len|nLines|nTessPoints)$/u.test(name)) return 64;
  if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize|size)$/u.test(name)) return 1024;
  if (/^(?:threads|threadsPerBlock|blockSize)$/u.test(name)) return 256;
  if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
  return 1;
}
