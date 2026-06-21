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
export type Cs149SimdInstructionOp = "load-values" | "load-exponents" | "add-accumulator" | "horizontal-add" | "mul" | "decrement-exponents" | "clamp" | "store";
export interface Cs149SimdTraceEvent {
    step: number;
    op: Cs149SimdInstructionOp;
    chunk: number;
    laneStart: number;
    mask: boolean[];
    activeLanes: number;
}
export interface SimdKernelStats {
    vectorWidth: number;
    chunks: number;
    inputLanes: number;
    laneSlots: number;
    tailInactiveLanes: number;
    vectorInstructions: number;
    activeLanes: number;
    totalLanes: number;
    utilization: number;
}
export interface Cs149ClampedExpVectorInput {
    readonly values: readonly number[];
    readonly exponents: readonly number[];
    readonly vectorWidth: number;
    readonly clamp?: number;
}
export interface Cs149ClampedExpVectorResult {
    output: number[];
    stats: SimdKernelStats;
    trace: Cs149SimdTraceEvent[];
}
export interface Cs149ArraySumVectorInput {
    readonly values: readonly number[];
    readonly vectorWidth: number;
}
export interface Cs149ArraySumVectorResult {
    sum: number;
    partialLaneSums: number[];
    horizontalReductionRounds: number;
    stats: SimdKernelStats;
    trace: Cs149SimdTraceEvent[];
}
export interface StaticWorkRange {
    start: number;
    end: number;
}
export interface StaticWorkPartition {
    worker: number;
    ranges: StaticWorkRange[];
}
export interface StaticWorkPartitionInput {
    readonly items: number;
    readonly workers: number;
    readonly chunkSize?: number;
}
export interface DistributedParameterSpec {
    readonly name: string;
    readonly trainable?: boolean;
}
export type RankGradientMap = Readonly<Record<string, readonly number[]>>;
export interface DdpGradientSyncEvent {
    readonly step: number;
    readonly kind: "ddp-gradient-all-reduce";
    readonly parameter: string;
    readonly participants: readonly number[];
    readonly inputGradients: readonly (readonly number[])[];
    readonly outputGradient: readonly number[];
}
export interface DdpGradientSynchronizationInput {
    readonly parameters: readonly DistributedParameterSpec[];
    readonly rankGradients: readonly RankGradientMap[];
}
export interface DdpGradientSynchronizationResult {
    readonly synchronizedGradients: readonly RankGradientMap[];
    readonly events: readonly DdpGradientSyncEvent[];
}
export interface DistributedParameterValueSpec {
    readonly name: string;
    readonly values: readonly number[];
    readonly sharded?: boolean;
}
export interface FsdpTensorShard {
    readonly start: number;
    readonly end: number;
    readonly values: readonly number[];
    readonly replicated?: boolean;
}
export type FsdpRankShardMap = Readonly<Record<string, FsdpTensorShard>>;
export interface FsdpParameterShardPlan {
    readonly name: string;
    readonly length: number;
    readonly sharded: boolean;
    readonly ranges: readonly {
        readonly rank: number;
        readonly start: number;
        readonly end: number;
        readonly replicated?: boolean;
    }[];
}
export interface FsdpShardPlan {
    readonly ranks: number;
    readonly parameters: readonly FsdpParameterShardPlan[];
}
export type FsdpShardingEventKind = "fsdp-shard" | "fsdp-replicate" | "fsdp-all-gather" | "fsdp-reduce-scatter" | "fsdp-replicated-gradient-all-reduce";
export interface FsdpShardingEvent {
    readonly step: number;
    readonly kind: FsdpShardingEventKind;
    readonly parameter?: string;
    readonly participants: readonly number[];
    readonly ranges?: readonly {
        readonly rank: number;
        readonly start: number;
        readonly end: number;
    }[];
    readonly output?: readonly number[];
}
export interface FsdpParameterShardingInput {
    readonly ranks: number;
    readonly parameters: readonly DistributedParameterValueSpec[];
}
export interface FsdpParameterShardingResult {
    readonly shardPlan: FsdpShardPlan;
    readonly rankShards: readonly FsdpRankShardMap[];
    readonly fullParameters: Readonly<Record<string, readonly number[]>>;
    readonly events: readonly FsdpShardingEvent[];
}
export interface FsdpGradientReduceScatterInput {
    readonly shardPlan: FsdpShardPlan;
    readonly rankGradients: readonly RankGradientMap[];
}
export interface FsdpGradientReduceScatterResult {
    readonly rankGradientShards: readonly FsdpRankShardMap[];
    readonly events: readonly FsdpShardingEvent[];
}
export interface ShardedAdamWParameterSpec {
    readonly name: string;
    readonly values: readonly number[];
    readonly gradients: readonly number[];
    readonly trainable?: boolean;
}
export interface ShardedAdamWOptions {
    readonly lr: number;
    readonly weightDecay?: number;
    readonly betas?: readonly [number, number];
    readonly eps?: number;
}
export interface ShardedAdamWState {
    readonly step: number;
    readonly expAvg: readonly number[];
    readonly expAvgSq: readonly number[];
}
export interface ShardedAdamWInput {
    readonly ranks: number;
    readonly parameters: readonly ShardedAdamWParameterSpec[];
    readonly optimizer: ShardedAdamWOptions;
    readonly state?: Readonly<Record<string, ShardedAdamWState>>;
}
export interface ShardedOptimizerOwnership {
    readonly rank: number;
    readonly parameters: readonly string[];
    readonly elements: number;
}
export type ShardedOptimizerEventKind = "optimizer-state-shard" | "sharded-adamw-step";
export interface ShardedOptimizerEvent {
    readonly step: number;
    readonly kind: ShardedOptimizerEventKind;
    readonly participants: readonly number[];
    readonly rank?: number;
    readonly parameters?: readonly string[];
}
export interface ShardedAdamWResult {
    readonly updatedParameters: Readonly<Record<string, readonly number[]>>;
    readonly nextState: Readonly<Record<string, ShardedAdamWState>>;
    readonly ownership: readonly ShardedOptimizerOwnership[];
    readonly events: readonly ShardedOptimizerEvent[];
}
export declare class SimulatorError extends Error {
    constructor(message: string);
}
export declare function createDeterministicMesh(options: DeterministicMeshOptions): DeterministicMesh;
export declare function simulateCs149ClampedExpVector(input: Cs149ClampedExpVectorInput): Cs149ClampedExpVectorResult;
export declare function simulateCs149ArraySumVector(input: Cs149ArraySumVectorInput): Cs149ArraySumVectorResult;
export declare function partitionStaticWork(input: StaticWorkPartitionInput): StaticWorkPartition[];
export declare function simulateShardedAdamWStep(input: ShardedAdamWInput): ShardedAdamWResult;
export declare function simulateDdpGradientSynchronization(input: DdpGradientSynchronizationInput): DdpGradientSynchronizationResult;
export declare function simulateFsdpParameterSharding(input: FsdpParameterShardingInput): FsdpParameterShardingResult;
export declare function simulateFsdpGradientReduceScatter(input: FsdpGradientReduceScatterInput): FsdpGradientReduceScatterResult;
export declare function createTaskGraphSimulator(options: TaskGraphSimulatorOptions): TaskGraphSimulator;
//# sourceMappingURL=index.d.ts.map