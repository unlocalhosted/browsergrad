declare const __BG_REQUIRE_WEBGPU__: boolean;

export interface WebGpuEvidenceDevice {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly adapterInfo: Readonly<{
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  }>;
}

export type WebGpuEvidenceAcquisition =
  | { readonly kind: "available"; readonly value: WebGpuEvidenceDevice }
  | { readonly kind: "unavailable"; readonly reason: string };

export function requiresWebGpuEvidence(): boolean {
  return __BG_REQUIRE_WEBGPU__;
}

/** Acquires the browser's real adapter/device path; injected devices are forbidden here. */
export async function acquireWebGpuEvidenceDevice(): Promise<WebGpuEvidenceAcquisition> {
  if (typeof navigator === "undefined" || navigator.gpu === undefined) {
    return unavailable("navigator.gpu is unavailable");
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (error) {
    return unavailable(`requestAdapter failed: ${message(error)}`);
  }
  if (adapter === null) return unavailable("requestAdapter returned no adapter");
  try {
    const device = await adapter.requestDevice();
    return {
      kind: "available",
      value: {
        adapter,
        device,
        adapterInfo: Object.freeze({
          vendor: adapter.info?.vendor ?? "",
          architecture: adapter.info?.architecture ?? "",
          device: adapter.info?.device ?? "",
          description: adapter.info?.description ?? "",
        }),
      },
    };
  } catch (error) {
    return unavailable(`requestDevice failed: ${message(error)}`);
  }
}

export function requiredEvidenceFailure(reason: string): Error {
  return new Error(`required WebGPU evidence unavailable: ${reason}`);
}

function unavailable(reason: string): WebGpuEvidenceAcquisition {
  return { kind: "unavailable", reason };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
