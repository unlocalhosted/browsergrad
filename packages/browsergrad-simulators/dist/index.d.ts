export type SimulationEventKind = "barrier" | "send" | "deliver" | "all-reduce";
export type SimulationReductionOp = "sum" | "min" | "max";
export interface SimulationEvent {
    readonly step: number;
    readonly kind: SimulationEventKind;
    readonly tag: string;
    readonly from?: number;
    readonly to?: number;
    readonly participants?: number[];
    readonly payload?: unknown;
    readonly op?: SimulationReductionOp;
    readonly values?: readonly number[];
    readonly result?: number;
}
export interface DeterministicMeshOptions {
    readonly ranks: number;
}
export interface MeshSendOptions {
    readonly from: number;
    readonly to: number;
    readonly tag: string;
    readonly payload?: unknown;
}
export interface MeshBroadcastOptions {
    readonly from: number;
    readonly tag: string;
    readonly payload?: unknown;
}
export interface MeshAllReduceOptions {
    readonly tag: string;
    readonly values: readonly number[];
    readonly op: SimulationReductionOp;
}
export interface DeterministicMesh {
    readonly rankCount: number;
    barrier(tag?: string): void;
    send(message: MeshSendOptions): void;
    broadcast(message: MeshBroadcastOptions): void;
    allReduce(options: MeshAllReduceOptions): number[];
    trace(): SimulationEvent[];
    clear(): void;
}
export declare class SimulatorError extends Error {
    constructor(message: string);
}
export declare function createDeterministicMesh(options: DeterministicMeshOptions): DeterministicMesh;
//# sourceMappingURL=index.d.ts.map