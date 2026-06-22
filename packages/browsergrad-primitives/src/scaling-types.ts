export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export interface DispatchExperiment {
  readonly id: number | string;
  readonly userId: string;
  readonly training_config: JsonObject;
  readonly status: {
    readonly status_type: string;
    readonly queued_at?: string;
  };
}

export interface DispatchSelection<T extends DispatchExperiment> {
  readonly experiments: readonly T[];
  readonly currentlyRunningJobs: number;
  readonly zeroRunningUserExperiments: number;
}

export interface DispatchSelectorOptions {
  readonly maxConcurrentWorkers: number;
}

export interface PowerLawScalingFit {
  readonly x: string;
  readonly y: string;
  readonly slope: number;
  readonly intercept: number;
  readonly exponent: number;
  readonly multiplier: number;
  readonly rSquared: number;
  predict(x: number): number;
}

export interface PowerLawFitOptions {
  readonly x: string;
  readonly y: string;
}
