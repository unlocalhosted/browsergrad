export class ScalingApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ScalingApiError";
    this.status = status;
    this.detail = detail;
  }
}
