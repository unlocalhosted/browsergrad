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
export type TaskGraphEventKind = "task-ready" | "task-start" | "task-finish";
export interface TaskSpec {
    readonly id: string;
    readonly duration: number;
    readonly dependsOn?: readonly string[];
}
export interface TaskGraphEvent {
    time: number;
    kind: TaskGraphEventKind;
    taskId: string;
    worker?: number;
}
export interface TaskGraphRunResult {
    readonly makespan: number;
    readonly completedTaskIds: string[];
    readonly events: TaskGraphEvent[];
}
export interface TaskGraphSimulatorOptions {
    readonly workers: number;
}
export interface TaskGraphSimulator {
    readonly workerCount: number;
    addTask(task: TaskSpec): void;
    run(): TaskGraphRunResult;
    clear(): void;
}
export declare class SimulatorError extends Error {
    constructor(message: string);
}
export declare function createDeterministicMesh(options: DeterministicMeshOptions): DeterministicMesh;
export declare function createTaskGraphSimulator(options: TaskGraphSimulatorOptions): TaskGraphSimulator;
//# sourceMappingURL=index.d.ts.map