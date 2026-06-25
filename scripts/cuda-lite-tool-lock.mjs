import { randomUUID } from "node:crypto";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const ownerFileName = "owner.json";

export async function withDirectoryLock(dir, options, fn) {
  const lock = await acquireDirectoryLock(dir, options);
  try {
    return await fn();
  } finally {
    lock.stopHeartbeat();
    await lock.release();
  }
}

export async function acquireDirectoryLock(dir, options = {}) {
  const {
    heartbeatMs = 5_000,
    sleepMs = defaultSleep,
    tool = "unknown",
  } = options;
  await mkdir(path.dirname(dir), { recursive: true });

  while (true) {
    const owner = createLockOwner({ tool, pid: options.pid, now: options.now });
    try {
      await mkdir(dir);
      await writeLockOwner(dir, owner);
      const interval = startHeartbeat(dir, owner, {
        heartbeatMs,
        now: options.now,
      });
      return {
        owner,
        token: owner.token,
        stopHeartbeat: () => clearInterval(interval),
        release: () => releaseDirectoryLock(dir, owner.token),
      };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      await removeStaleLock(dir, options);
      await sleepMs(75);
    }
  }
}

export function createLockOwner({ tool = "unknown", pid = process.pid, now = Date.now } = {}) {
  const timestamp = new Date(now()).toISOString();
  return {
    pid,
    token: randomUUID(),
    tool,
    createdAt: timestamp,
    heartbeatAt: timestamp,
  };
}

export async function releaseDirectoryLock(dir, token) {
  const owner = await readLockOwner(dir);
  if (owner?.token !== token) return false;
  await rm(dir, { recursive: true, force: true });
  return true;
}

export async function removeStaleLock(dir, options = {}) {
  const {
    isProcessAlive = defaultIsProcessAlive,
    now = Date.now,
    staleLockMs = 30 * 60 * 1000,
  } = options;
  try {
    const owner = await readLockOwner(dir);
    if (owner !== undefined) {
      const heartbeatMs = Date.parse(owner.heartbeatAt);
      const heartbeatFresh = Number.isFinite(heartbeatMs) && now() - heartbeatMs <= staleLockMs;
      if (heartbeatFresh && isProcessAlive(owner.pid)) return false;
      await rm(dir, { recursive: true, force: true });
      return true;
    }
    const info = await stat(dir);
    if (now() - info.mtimeMs > staleLockMs) {
      await rm(dir, { recursive: true, force: true });
      return true;
    }
    return false;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

export async function readLockOwner(dir) {
  try {
    const raw = await readFile(path.join(dir, ownerFileName), "utf8");
    const owner = JSON.parse(raw);
    if (!Number.isInteger(owner?.pid) || owner.pid <= 0) return undefined;
    if (typeof owner.token !== "string" || owner.token.length === 0) return undefined;
    if (typeof owner.heartbeatAt !== "string" || owner.heartbeatAt.length === 0) return undefined;
    return {
      pid: owner.pid,
      token: owner.token,
      tool: typeof owner.tool === "string" ? owner.tool : "unknown",
      createdAt: typeof owner.createdAt === "string" ? owner.createdAt : owner.heartbeatAt,
      heartbeatAt: owner.heartbeatAt,
    };
  } catch {
    return undefined;
  }
}

export async function writeLockOwner(dir, owner) {
  await writeFile(path.join(dir, ownerFileName), `${JSON.stringify(owner)}\n`);
}

async function heartbeatLockOwner(dir, owner, now = Date.now) {
  const current = await readLockOwner(dir);
  if (current?.token !== owner.token) return false;
  await writeLockOwner(dir, {
    ...owner,
    heartbeatAt: new Date(now()).toISOString(),
  });
  return true;
}

function startHeartbeat(dir, owner, options) {
  const { heartbeatMs, now } = options;
  const interval = setInterval(() => {
    void heartbeatLockOwner(dir, owner, now).catch(() => {
      clearInterval(interval);
    });
  }, heartbeatMs);
  interval.unref?.();
  return interval;
}

function defaultIsProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === "EPERM";
  }
}

function defaultSleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
