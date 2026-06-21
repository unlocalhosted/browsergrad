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
function validateRankCount(ranks) {
    if (!Number.isInteger(ranks) || ranks <= 0) {
        throw new SimulatorError("ranks must be a positive integer");
    }
    return ranks;
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
//# sourceMappingURL=index.js.map