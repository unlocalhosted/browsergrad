export class SimulatorError extends Error {
    constructor(message) {
        super(message);
        this.name = "SimulatorError";
    }
}
export function createDeterministicMesh(options) {
    const rankCount = validateRankCount(options.ranks);
    const events = [];
    let step = 0;
    const push = (event) => {
        events.push({ step, ...cloneEvent(event) });
        step += 1;
    };
    const validateRank = (rank, field) => {
        if (!Number.isInteger(rank) || rank < 0 || rank >= rankCount) {
            throw new SimulatorError(`${field} must be an integer rank in [0, ${rankCount})`);
        }
    };
    const send = (message) => {
        validateRank(message.from, "from");
        validateRank(message.to, "to");
        if (message.from === message.to) {
            throw new SimulatorError("send requires distinct from/to ranks");
        }
        push({
            kind: "send",
            tag: message.tag,
            from: message.from,
            to: message.to,
            ...(message.payload !== undefined ? { payload: message.payload } : {}),
        });
        push({
            kind: "deliver",
            tag: message.tag,
            from: message.from,
            to: message.to,
            ...(message.payload !== undefined ? { payload: message.payload } : {}),
        });
    };
    return {
        rankCount,
        barrier(tag = "barrier") {
            push({
                kind: "barrier",
                tag,
                participants: allRanks(rankCount),
            });
        },
        send,
        broadcast(message) {
            validateRank(message.from, "from");
            for (const to of allRanks(rankCount)) {
                if (to === message.from)
                    continue;
                send({
                    from: message.from,
                    to,
                    tag: message.tag,
                    ...(message.payload !== undefined ? { payload: message.payload } : {}),
                });
            }
        },
        allReduce(options) {
            if (options.values.length !== rankCount) {
                throw new SimulatorError(`allReduce values length must equal rank count ${rankCount}`);
            }
            for (let i = 0; i < options.values.length; i++) {
                const value = options.values[i];
                if (typeof value !== "number" || !Number.isFinite(value)) {
                    throw new SimulatorError(`allReduce values[${i}] must be finite`);
                }
            }
            const result = reduceValues(options.values, options.op);
            push({
                kind: "all-reduce",
                tag: options.tag,
                op: options.op,
                participants: allRanks(rankCount),
                values: [...options.values],
                result,
            });
            return Array.from({ length: rankCount }, () => result);
        },
        trace() {
            return events.map(cloneEvent);
        },
        clear() {
            events.length = 0;
            step = 0;
        },
    };
}
export function simulateCs149ClampedExpVector(input) {
    const vectorWidth = validateVectorWidth(input.vectorWidth);
    const values = validateNumberVector(input.values, "values");
    if (values.length === 0) {
        throw new SimulatorError("values must contain at least one lane");
    }
    const exponents = validateExponentVector(input.exponents);
    if (exponents.length !== values.length) {
        throw new SimulatorError("values and exponents must have the same length");
    }
    const clamp = finitePositiveNumber(input.clamp ?? 9.999999, "clamp");
    const chunks = Math.ceil(values.length / vectorWidth);
    const laneSlots = chunks * vectorWidth;
    const output = Array.from({ length: values.length }, () => 0);
    const trace = [];
    let step = 0;
    let activeLaneTotal = 0;
    let laneTotal = 0;
    const record = (op, chunk, laneStart, mask) => {
        const activeLanes = countActive(mask);
        trace.push({
            step,
            op,
            chunk,
            laneStart,
            mask: [...mask],
            activeLanes,
        });
        step += 1;
        activeLaneTotal += activeLanes;
        laneTotal += vectorWidth;
    };
    for (let chunk = 0; chunk < chunks; chunk++) {
        const laneStart = chunk * vectorWidth;
        const validMask = Array.from({ length: vectorWidth }, (_, lane) => laneStart + lane < values.length);
        const result = Array.from({ length: vectorWidth }, () => 1);
        const remaining = Array.from({ length: vectorWidth }, (_, lane) => {
            const index = laneStart + lane;
            return validMask[lane] ? exponents[index] : 0;
        });
        record("load-values", chunk, laneStart, validMask);
        record("load-exponents", chunk, laneStart, validMask);
        while (remaining.some((count, lane) => validMask[lane] === true && count > 0)) {
            const activeMask = remaining.map((count, lane) => validMask[lane] === true && count > 0);
            record("mul", chunk, laneStart, activeMask);
            for (let lane = 0; lane < vectorWidth; lane++) {
                if (!activeMask[lane])
                    continue;
                result[lane] *= values[laneStart + lane];
                remaining[lane] -= 1;
            }
            record("decrement-exponents", chunk, laneStart, activeMask);
        }
        const clampMask = result.map((value, lane) => validMask[lane] === true && value > clamp);
        record("clamp", chunk, laneStart, clampMask);
        for (let lane = 0; lane < vectorWidth; lane++) {
            const index = laneStart + lane;
            if (!validMask[lane] || index >= output.length)
                continue;
            output[index] = clampMask[lane] ? clamp : result[lane];
        }
        record("store", chunk, laneStart, validMask);
    }
    return {
        output: [...output],
        stats: {
            vectorWidth,
            chunks,
            inputLanes: values.length,
            laneSlots,
            tailInactiveLanes: laneSlots - values.length,
            vectorInstructions: trace.length,
            activeLanes: activeLaneTotal,
            totalLanes: laneTotal,
            utilization: laneTotal === 0 ? 0 : activeLaneTotal / laneTotal,
        },
        trace: trace.map(cloneCs149SimdTraceEvent),
    };
}
export function simulateCs149ArraySumVector(input) {
    const vectorWidth = validateVectorWidth(input.vectorWidth);
    const values = validateNumberVector(input.values, "values");
    if (values.length === 0) {
        throw new SimulatorError("values must contain at least one lane");
    }
    if (values.length % vectorWidth !== 0) {
        throw new SimulatorError("arraySumVector fixtures must have values.length divisible by vectorWidth");
    }
    const chunks = values.length / vectorWidth;
    const laneSlots = values.length;
    const partialLaneSums = Array.from({ length: vectorWidth }, () => 0);
    const trace = [];
    let step = 0;
    let activeLaneTotal = 0;
    let laneTotal = 0;
    const record = (op, chunk, laneStart, mask) => {
        const activeLanes = countActive(mask);
        trace.push({
            step,
            op,
            chunk,
            laneStart,
            mask: [...mask],
            activeLanes,
        });
        step += 1;
        activeLaneTotal += activeLanes;
        laneTotal += vectorWidth;
    };
    const fullMask = Array.from({ length: vectorWidth }, () => true);
    for (let chunk = 0; chunk < chunks; chunk++) {
        const laneStart = chunk * vectorWidth;
        record("load-values", chunk, laneStart, fullMask);
        record("add-accumulator", chunk, laneStart, fullMask);
        for (let lane = 0; lane < vectorWidth; lane++) {
            partialLaneSums[lane] += values[laneStart + lane];
        }
    }
    let working = [...partialLaneSums];
    let horizontalReductionRounds = 0;
    while (working.length > 1) {
        const active = working.length;
        record("horizontal-add", chunks, 0, Array.from({ length: vectorWidth }, (_, lane) => lane < active));
        const next = [];
        for (let lane = 0; lane < working.length; lane += 2) {
            next.push(working[lane] + (working[lane + 1] ?? 0));
        }
        working = next;
        horizontalReductionRounds += 1;
    }
    return {
        sum: working[0] ?? 0,
        partialLaneSums: [...partialLaneSums],
        horizontalReductionRounds,
        stats: {
            vectorWidth,
            chunks,
            inputLanes: values.length,
            laneSlots,
            tailInactiveLanes: 0,
            vectorInstructions: trace.length,
            activeLanes: activeLaneTotal,
            totalLanes: laneTotal,
            utilization: laneTotal === 0 ? 0 : activeLaneTotal / laneTotal,
        },
        trace: trace.map(cloneCs149SimdTraceEvent),
    };
}
export function partitionStaticWork(input) {
    const itemCount = validateItemCount(input.items);
    const workerCount = validateRankCount(input.workers);
    const partitions = Array.from({ length: workerCount }, (_, worker) => ({
        worker,
        ranges: [],
    }));
    if (input.chunkSize !== undefined) {
        const chunkSize = validateChunkSize(input.chunkSize);
        let chunkIndex = 0;
        for (let start = 0; start < itemCount; start += chunkSize) {
            const worker = chunkIndex % workerCount;
            partitions[worker].ranges.push({
                start,
                end: Math.min(start + chunkSize, itemCount),
            });
            chunkIndex += 1;
        }
        return partitions.map(cloneStaticWorkPartition);
    }
    const base = Math.floor(itemCount / workerCount);
    const remainder = itemCount % workerCount;
    let start = 0;
    for (let worker = 0; worker < workerCount; worker++) {
        const size = base + (worker < remainder ? 1 : 0);
        if (size > 0) {
            partitions[worker].ranges.push({
                start,
                end: start + size,
            });
        }
        start += size;
    }
    return partitions.map(cloneStaticWorkPartition);
}
export function simulateShardedAdamWStep(input) {
    const rankCount = validateRankCount(input.ranks);
    const options = normalizeAdamWOptions(input.optimizer);
    const ownership = assignOptimizerStateOwners(input.parameters, rankCount);
    const updatedParameters = {};
    const nextState = {};
    const events = [];
    let eventStep = 0;
    for (const owner of ownership) {
        events.push({
            step: eventStep,
            kind: "optimizer-state-shard",
            participants: allRanks(rankCount),
            rank: owner.rank,
            parameters: [...owner.parameters],
        });
        eventStep += 1;
    }
    for (const parameter of input.parameters) {
        validateParameterName(parameter.name);
        const values = validateTensorValues(parameter.values, parameter.name);
        if (parameter.trainable === false) {
            updatedParameters[parameter.name] = values;
            continue;
        }
        const gradients = validateTensorValues(parameter.gradients, `${parameter.name}.grad`);
        if (gradients.length !== values.length) {
            throw new SimulatorError(`gradient length for ${parameter.name} must match parameter length`);
        }
        const previous = normalizeAdamWState(input.state?.[parameter.name], values.length, parameter.name);
        const stepped = adamWStep(values, gradients, previous, options);
        updatedParameters[parameter.name] = stepped.values;
        nextState[parameter.name] = stepped.state;
    }
    events.push({
        step: eventStep,
        kind: "sharded-adamw-step",
        participants: allRanks(rankCount),
    });
    return {
        updatedParameters: cloneFullParameterMap(updatedParameters),
        nextState: cloneAdamWStateMap(nextState),
        ownership: ownership.map(cloneShardedOptimizerOwnership),
        events: events.map(cloneShardedOptimizerEvent),
    };
}
export function simulateDdpGradientSynchronization(input) {
    const rankCount = validateRankCount(input.rankGradients.length);
    const synchronizedGradients = Array.from({ length: rankCount }, () => ({}));
    const events = [];
    let step = 0;
    for (const parameter of input.parameters) {
        validateParameterName(parameter.name);
        if (parameter.trainable === false)
            continue;
        const rankValues = input.rankGradients.map((gradients, rank) => readGradient(gradients, parameter.name, rank));
        const outputGradient = averageGradients(rankValues, parameter.name);
        for (const rankGradients of synchronizedGradients) {
            rankGradients[parameter.name] = [...outputGradient];
        }
        events.push({
            step,
            kind: "ddp-gradient-all-reduce",
            parameter: parameter.name,
            participants: allRanks(rankCount),
            inputGradients: rankValues.map((values) => [...values]),
            outputGradient: [...outputGradient],
        });
        step += 1;
    }
    return {
        synchronizedGradients: synchronizedGradients.map(cloneRankGradientMap),
        events: events.map(cloneDdpGradientSyncEvent),
    };
}
export function simulateFsdpParameterSharding(input) {
    const rankCount = validateRankCount(input.ranks);
    const rankShards = createEmptyRankShardMaps(rankCount);
    const fullParameters = {};
    const parameters = [];
    const events = [];
    let step = 0;
    for (const parameter of input.parameters) {
        validateParameterName(parameter.name);
        const values = validateTensorValues(parameter.values, parameter.name);
        const sharded = parameter.sharded !== false;
        const ranges = sharded
            ? balancedRanges(values.length, rankCount).map((range, rank) => ({
                rank,
                start: range.start,
                end: range.end,
            }))
            : allRanks(rankCount).map((rank) => ({
                rank,
                start: 0,
                end: values.length,
                replicated: true,
            }));
        for (const range of ranges) {
            rankShards[range.rank][parameter.name] = {
                start: range.start,
                end: range.end,
                values: values.slice(range.start, range.end),
                ...(range.replicated ? { replicated: true } : {}),
            };
        }
        fullParameters[parameter.name] = [...values];
        parameters.push({
            name: parameter.name,
            length: values.length,
            sharded,
            ranges,
        });
        events.push({
            step,
            kind: sharded ? "fsdp-shard" : "fsdp-replicate",
            parameter: parameter.name,
            participants: allRanks(rankCount),
            ranges: ranges.map(({ rank, start, end }) => ({ rank, start, end })),
        });
        step += 1;
    }
    events.push({
        step,
        kind: "fsdp-all-gather",
        participants: allRanks(rankCount),
    });
    return {
        shardPlan: {
            ranks: rankCount,
            parameters: parameters.map(cloneFsdpParameterShardPlan),
        },
        rankShards: rankShards.map(cloneFsdpRankShardMap),
        fullParameters: cloneFullParameterMap(fullParameters),
        events: events.map(cloneFsdpShardingEvent),
    };
}
export function simulateFsdpGradientReduceScatter(input) {
    const rankCount = validateRankCount(input.shardPlan.ranks);
    if (input.rankGradients.length !== rankCount) {
        throw new SimulatorError("rankGradients length must match shardPlan ranks");
    }
    const rankGradientShards = createEmptyRankShardMaps(rankCount);
    const events = [];
    let step = 0;
    for (const parameter of input.shardPlan.parameters) {
        const rankValues = input.rankGradients.map((gradients, rank) => readGradient(gradients, parameter.name, rank));
        const averaged = averageGradients(rankValues, parameter.name);
        if (averaged.length !== parameter.length) {
            throw new SimulatorError(`gradient length for parameter ${parameter.name} does not match shard plan`);
        }
        for (const range of parameter.ranges) {
            rankGradientShards[range.rank][parameter.name] = {
                start: range.start,
                end: range.end,
                values: averaged.slice(range.start, range.end),
                ...(range.replicated ? { replicated: true } : {}),
            };
        }
        events.push({
            step,
            kind: parameter.sharded
                ? "fsdp-reduce-scatter"
                : "fsdp-replicated-gradient-all-reduce",
            parameter: parameter.name,
            participants: allRanks(rankCount),
            ranges: parameter.ranges.map(({ rank, start, end }) => ({ rank, start, end })),
            output: [...averaged],
        });
        step += 1;
    }
    return {
        rankGradientShards: rankGradientShards.map(cloneFsdpRankShardMap),
        events: events.map(cloneFsdpShardingEvent),
    };
}
function validateRankCount(ranks) {
    if (!Number.isInteger(ranks) || ranks <= 0) {
        throw new SimulatorError("ranks must be a positive integer");
    }
    return ranks;
}
function validateVectorWidth(width) {
    if (!Number.isInteger(width) || width <= 0) {
        throw new SimulatorError("vectorWidth must be a positive integer");
    }
    return width;
}
function validateNumberVector(values, name) {
    return values.map((value, index) => {
        if (!Number.isFinite(value)) {
            throw new SimulatorError(`${name}[${index}] must be finite`);
        }
        return value;
    });
}
function validateExponentVector(exponents) {
    return exponents.map((value, index) => {
        if (!Number.isInteger(value) || value < 0) {
            throw new SimulatorError(`exponents[${index}] must be a non-negative integer`);
        }
        return value;
    });
}
function countActive(mask) {
    return mask.reduce((count, active) => count + (active ? 1 : 0), 0);
}
function validateItemCount(items) {
    if (!Number.isInteger(items) || items < 0) {
        throw new SimulatorError("items must be a non-negative integer");
    }
    return items;
}
function validateChunkSize(chunkSize) {
    if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
        throw new SimulatorError("chunkSize must be a positive integer");
    }
    return chunkSize;
}
function cloneCs149SimdTraceEvent(event) {
    return {
        ...event,
        mask: [...event.mask],
    };
}
function cloneStaticWorkPartition(partition) {
    return {
        worker: partition.worker,
        ranges: partition.ranges.map((range) => ({ ...range })),
    };
}
function allRanks(rankCount) {
    return Array.from({ length: rankCount }, (_, rank) => rank);
}
function reduceValues(values, op) {
    if (op === "sum") {
        return values.reduce((acc, value) => acc + value, 0);
    }
    if (op === "min") {
        return Math.min(...values);
    }
    if (op === "max") {
        return Math.max(...values);
    }
    throw new SimulatorError(`unsupported reduction op: ${String(op)}`);
}
function cloneEvent(event) {
    return {
        ...event,
        ...(event.participants ? { participants: [...event.participants] } : {}),
        ...(event.values ? { values: [...event.values] } : {}),
        ...(event.payload !== undefined ? { payload: clonePayload(event.payload) } : {}),
    };
}
function clonePayload(payload) {
    if (payload === null || typeof payload !== "object")
        return payload;
    if (Array.isArray(payload))
        return payload.map(clonePayload);
    if (payload instanceof Uint8Array)
        return new Uint8Array(payload);
    const out = {};
    for (const [key, value] of Object.entries(payload)) {
        out[key] = clonePayload(value);
    }
    return out;
}
function validateParameterName(name) {
    if (name.length === 0) {
        throw new SimulatorError("parameter name must be non-empty");
    }
}
function readGradient(gradients, parameter, rank) {
    const value = gradients[parameter];
    if (!value) {
        throw new SimulatorError(`rank ${rank} is missing gradient for parameter ${parameter}`);
    }
    return value.map((entry, index) => {
        if (!Number.isFinite(entry)) {
            throw new SimulatorError(`gradient ${parameter}[${index}] on rank ${rank} must be finite`);
        }
        return entry;
    });
}
function averageGradients(gradients, parameter) {
    const width = gradients[0]?.length;
    if (width === undefined) {
        throw new SimulatorError(`parameter ${parameter} has no gradient values`);
    }
    for (let rank = 0; rank < gradients.length; rank++) {
        if ((gradients[rank]?.length ?? -1) !== width) {
            throw new SimulatorError(`gradient shape mismatch for parameter ${parameter} on rank ${rank}`);
        }
    }
    return Array.from({ length: width }, (_, index) => {
        const sum = gradients.reduce((total, values) => total + (values[index] ?? 0), 0);
        return sum / gradients.length;
    });
}
function cloneRankGradientMap(map) {
    const out = {};
    for (const [name, values] of Object.entries(map)) {
        out[name] = [...values];
    }
    return out;
}
function cloneDdpGradientSyncEvent(event) {
    return {
        ...event,
        participants: [...event.participants],
        inputGradients: event.inputGradients.map((gradient) => [...gradient]),
        outputGradient: [...event.outputGradient],
    };
}
function validateTensorValues(values, name) {
    if (values.length === 0) {
        throw new SimulatorError(`parameter ${name} must have at least one value`);
    }
    return values.map((value, index) => {
        if (!Number.isFinite(value)) {
            throw new SimulatorError(`parameter ${name}[${index}] must be finite`);
        }
        return value;
    });
}
function normalizeAdamWOptions(options) {
    const lr = finitePositiveNumber(options.lr, "lr");
    const weightDecay = finiteNonNegativeNumber(options.weightDecay ?? 0, "weightDecay");
    const betas = options.betas ?? [0.9, 0.999];
    const beta1 = finiteBeta(betas[0], "beta1");
    const beta2 = finiteBeta(betas[1], "beta2");
    const eps = finiteNonNegativeNumber(options.eps ?? 1e-8, "eps");
    return { lr, weightDecay, betas: [beta1, beta2], eps };
}
function assignOptimizerStateOwners(parameters, rankCount) {
    const ownership = allRanks(rankCount).map((rank) => ({
        rank,
        parameters: [],
        elements: 0,
    }));
    for (const parameter of parameters) {
        validateParameterName(parameter.name);
        if (parameter.trainable === false)
            continue;
        const values = validateTensorValues(parameter.values, parameter.name);
        const owner = ownership.reduce((best, candidate) => {
            if (candidate.elements < best.elements)
                return candidate;
            if (candidate.elements === best.elements && candidate.rank < best.rank) {
                return candidate;
            }
            return best;
        });
        owner.parameters.push(parameter.name);
        owner.elements += values.length;
    }
    return ownership;
}
function normalizeAdamWState(state, length, parameter) {
    if (!state) {
        return {
            step: 0,
            expAvg: Array.from({ length }, () => 0),
            expAvgSq: Array.from({ length }, () => 0),
        };
    }
    if (!Number.isInteger(state.step) || state.step < 0) {
        throw new SimulatorError(`AdamW state step for ${parameter} must be non-negative`);
    }
    if (state.expAvg.length !== length || state.expAvgSq.length !== length) {
        throw new SimulatorError(`AdamW state for ${parameter} has wrong shape`);
    }
    return {
        step: state.step,
        expAvg: validateTensorValues(state.expAvg, `${parameter}.expAvg`),
        expAvgSq: validateTensorValues(state.expAvgSq, `${parameter}.expAvgSq`),
    };
}
function adamWStep(values, gradients, previous, options) {
    const [beta1, beta2] = options.betas;
    const step = previous.step + 1;
    const biasCorrection1 = 1 - beta1 ** step;
    const biasCorrection2 = 1 - beta2 ** step;
    const nextValues = [];
    const expAvg = [];
    const expAvgSq = [];
    for (let index = 0; index < values.length; index++) {
        const gradient = gradients[index] ?? 0;
        const nextExpAvg = beta1 * (previous.expAvg[index] ?? 0) + (1 - beta1) * gradient;
        const nextExpAvgSq = beta2 * (previous.expAvgSq[index] ?? 0) + (1 - beta2) * gradient ** 2;
        const correctedExpAvg = nextExpAvg / biasCorrection1;
        const correctedExpAvgSq = nextExpAvgSq / biasCorrection2;
        const decayedValue = (values[index] ?? 0) * (1 - options.lr * options.weightDecay);
        nextValues.push(decayedValue -
            (options.lr * correctedExpAvg) / (Math.sqrt(correctedExpAvgSq) + options.eps));
        expAvg.push(nextExpAvg);
        expAvgSq.push(nextExpAvgSq);
    }
    return {
        values: nextValues,
        state: { step, expAvg, expAvgSq },
    };
}
function finitePositiveNumber(value, name) {
    if (!Number.isFinite(value) || value <= 0) {
        throw new SimulatorError(`${name} must be positive`);
    }
    return value;
}
function finiteNonNegativeNumber(value, name) {
    if (!Number.isFinite(value) || value < 0) {
        throw new SimulatorError(`${name} must be non-negative`);
    }
    return value;
}
function finiteBeta(value, name) {
    if (!Number.isFinite(value) || value < 0 || value >= 1) {
        throw new SimulatorError(`${name} must be in [0, 1)`);
    }
    return value;
}
function cloneAdamWStateMap(map) {
    const out = {};
    for (const [name, state] of Object.entries(map)) {
        out[name] = {
            step: state.step,
            expAvg: [...state.expAvg],
            expAvgSq: [...state.expAvgSq],
        };
    }
    return out;
}
function cloneShardedOptimizerOwnership(ownership) {
    return {
        rank: ownership.rank,
        parameters: [...ownership.parameters],
        elements: ownership.elements,
    };
}
function cloneShardedOptimizerEvent(event) {
    return {
        ...event,
        participants: [...event.participants],
        ...(event.parameters ? { parameters: [...event.parameters] } : {}),
    };
}
function balancedRanges(length, rankCount) {
    const base = Math.floor(length / rankCount);
    const remainder = length % rankCount;
    const ranges = [];
    let start = 0;
    for (let rank = 0; rank < rankCount; rank++) {
        const width = base + (rank < remainder ? 1 : 0);
        ranges.push({ start, end: start + width });
        start += width;
    }
    return ranges;
}
function createEmptyRankShardMaps(rankCount) {
    return Array.from({ length: rankCount }, () => ({}));
}
function cloneFsdpRankShardMap(map) {
    const out = {};
    for (const [name, shard] of Object.entries(map)) {
        out[name] = {
            start: shard.start,
            end: shard.end,
            values: [...shard.values],
            ...(shard.replicated ? { replicated: true } : {}),
        };
    }
    return out;
}
function cloneFullParameterMap(map) {
    const out = {};
    for (const [name, values] of Object.entries(map)) {
        out[name] = [...values];
    }
    return out;
}
function cloneFsdpParameterShardPlan(plan) {
    return {
        ...plan,
        ranges: plan.ranges.map((range) => ({ ...range })),
    };
}
function cloneFsdpShardingEvent(event) {
    return {
        ...event,
        participants: [...event.participants],
        ...(event.ranges ? { ranges: event.ranges.map((range) => ({ ...range })) } : {}),
        ...(event.output ? { output: [...event.output] } : {}),
    };
}
export function createTaskGraphSimulator(options) {
    const workerCount = validateWorkerCount(options.workers);
    const tasks = new Map();
    return {
        workerCount,
        addTask(task) {
            const normalized = normalizeTask(task);
            if (tasks.has(normalized.id)) {
                throw new SimulatorError(`duplicate task id: ${normalized.id}`);
            }
            tasks.set(normalized.id, normalized);
        },
        run() {
            return runTaskGraph([...tasks.values()], workerCount);
        },
        clear() {
            tasks.clear();
        },
    };
}
function validateWorkerCount(workers) {
    if (!Number.isInteger(workers) || workers <= 0) {
        throw new SimulatorError("workers must be a positive integer");
    }
    return workers;
}
function normalizeTask(task) {
    if (task.id.length === 0) {
        throw new SimulatorError("task id must be non-empty");
    }
    if (!Number.isFinite(task.duration) || task.duration <= 0) {
        throw new SimulatorError(`task duration must be positive: ${task.id}`);
    }
    return {
        id: task.id,
        duration: task.duration,
        dependsOn: uniqueSorted(task.dependsOn ?? []),
    };
}
function runTaskGraph(tasks, workerCount) {
    const taskById = new Map(tasks.map((task) => [task.id, task]));
    validateTaskDependencies(tasks, taskById);
    const events = [];
    const completed = new Set();
    const started = new Set();
    const announcedReady = new Set();
    const completedTaskIds = [];
    const workers = Array.from({ length: workerCount }, (_, worker) => ({
        worker,
        taskId: undefined,
        finishTime: Number.POSITIVE_INFINITY,
    }));
    let time = 0;
    const emit = (event) => {
        events.push(cloneTaskGraphEvent(event));
    };
    while (completed.size < tasks.length) {
        const ready = readyTasks(tasks, completed, started);
        for (const task of ready) {
            if (!announcedReady.has(task.id)) {
                emit({ time, kind: "task-ready", taskId: task.id });
                announcedReady.add(task.id);
            }
        }
        let startedThisTick = false;
        for (const worker of workers.filter((item) => item.taskId === undefined)) {
            const next = ready.find((task) => !started.has(task.id));
            if (!next)
                break;
            started.add(next.id);
            worker.taskId = next.id;
            worker.finishTime = time + next.duration;
            emit({
                time,
                kind: "task-start",
                taskId: next.id,
                worker: worker.worker,
            });
            startedThisTick = true;
        }
        if (startedThisTick)
            continue;
        const nextFinishTime = Math.min(...workers.map((worker) => worker.finishTime));
        if (!Number.isFinite(nextFinishTime)) {
            throw new SimulatorError("task graph has a dependency cycle");
        }
        time = nextFinishTime;
        for (const worker of workers
            .filter((item) => item.finishTime === time && item.taskId !== undefined)
            .sort((a, b) => a.worker - b.worker)) {
            const taskId = worker.taskId;
            worker.taskId = undefined;
            worker.finishTime = Number.POSITIVE_INFINITY;
            completed.add(taskId);
            completedTaskIds.push(taskId);
            emit({ time, kind: "task-finish", taskId, worker: worker.worker });
        }
    }
    return {
        makespan: time,
        completedTaskIds: [...completedTaskIds],
        events: events.map(cloneTaskGraphEvent),
    };
}
function validateTaskDependencies(tasks, taskById) {
    for (const task of tasks) {
        for (const dependency of task.dependsOn) {
            if (!taskById.has(dependency)) {
                throw new SimulatorError(`task ${task.id} depends on missing task: ${dependency}`);
            }
        }
    }
}
function readyTasks(tasks, completed, started) {
    return tasks
        .filter((task) => !completed.has(task.id) &&
        !started.has(task.id) &&
        task.dependsOn.every((dependency) => completed.has(dependency)))
        .sort((a, b) => a.id.localeCompare(b.id));
}
function cloneTaskGraphEvent(event) {
    return { ...event };
}
function uniqueSorted(values) {
    return [...new Set(values)].sort();
}
//# sourceMappingURL=index.js.map