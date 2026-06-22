import { dispatch } from "./runner.js";
import type { KernelDevice } from "./types.js";
import type { Cuda1DProgram, Cuda1DProgramRunInput, Kernel1DProgram, Kernel1DProgramRunInput } from "./cuda_program-types.js";
import { emitKernel1DProgramWgsl } from "./cuda_program-wgsl.js";
import { materializeVector, readParameter } from "./cuda_program-validate.js";

export async function runCuda1DProgramWebGpu(
  device: KernelDevice,
  program: Cuda1DProgram,
  input: Cuda1DProgramRunInput = {},
): Promise<Float32Array> {
  return runKernel1DProgramWebGpu(device, program, input);
}

export async function runKernel1DProgramWebGpu(
  device: KernelDevice,
  program: Kernel1DProgram,
  input: Kernel1DProgramRunInput = {},
): Promise<Float32Array> {
  const initialInput = Float32Array.from(
    materializeVector(input.initialInput, program.inputLength, "initialInput"),
  );
  const initialOutput =
    input.initialOutput === undefined
      ? undefined
      : Float32Array.from(
          materializeVector(input.initialOutput, program.outputLength, "initialOutput"),
        );
  const parameterNames = Object.keys(program.parameters ?? {}).sort();
  const params = Float32Array.from(
    parameterNames.map((name) => readParameter(program.parameters ?? {}, name)),
  );

  return dispatch(
    device,
    {
      name: `kernel-1d-program-${program.name}`,
      wgsl: emitKernel1DProgramWgsl(program),
      workgroupSize: [program.launch.threadsPerBlock, 1, 1],
    },
    {
      inputs: [initialInput],
      outputLength: program.outputLength,
      params,
      ...(initialOutput ? { initialOutput } : {}),
      dispatchCount: [
        program.launch.blocks * program.launch.threadsPerBlock,
        1,
        1,
      ],
      cacheKeySuffix: [
        program.name,
        program.inputLength,
        program.outputLength,
        program.launch.blocks,
        program.launch.threadsPerBlock,
        parameterNames.join(","),
      ].join(":"),
    },
  );
}
