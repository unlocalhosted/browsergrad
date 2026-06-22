import { KernelError } from "./types.js";
import type { Cuda1DProgram, Cuda1DProgramInput, Kernel1DProgram, Kernel1DProgramInput } from "./cuda_program-types.js";
import { cloneStatement, validateIdentifier, validateNonNegativeInteger, validateParameters, validatePositiveInteger } from "./cuda_program-validate.js";

export type * from "./cuda_program-types.js";
export {
  runKernel1DProgramReference,
  simulateCuda1DProgram,
} from "./cuda_program-reference.js";
export { emitCuda1DProgramWgsl, emitKernel1DProgramWgsl } from "./cuda_program-wgsl.js";
export { runCuda1DProgramWebGpu, runKernel1DProgramWebGpu } from "./cuda_program-webgpu.js";

export function defineKernel1DProgram(
  input: Kernel1DProgramInput,
): Kernel1DProgram {
  return defineCuda1DProgram(input);
}

export function defineCuda1DProgram(input: Cuda1DProgramInput): Cuda1DProgram {
  validateIdentifier(input.name, "name");
  validateNonNegativeInteger(input.inputLength, "inputLength");
  validateNonNegativeInteger(input.outputLength, "outputLength");
  validatePositiveInteger(input.launch.blocks, "launch.blocks");
  validatePositiveInteger(
    input.launch.threadsPerBlock,
    "launch.threadsPerBlock",
  );
  if (input.body.length === 0) {
    throw new KernelError("Cuda1DProgram body must contain at least one statement");
  }
  const parameters = validateParameters(input.parameters ?? {});
  return {
    name: input.name,
    inputLength: input.inputLength,
    outputLength: input.outputLength,
    parameters,
    launch: {
      blocks: input.launch.blocks,
      threadsPerBlock: input.launch.threadsPerBlock,
    },
    body: input.body.map(cloneStatement),
  };
}
