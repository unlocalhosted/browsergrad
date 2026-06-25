#!/usr/bin/env node
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  acquireDirectoryLock,
  readLockOwner,
  releaseDirectoryLock,
  removeStaleLock,
  writeLockOwner,
} from "./cuda-lite-tool-lock.mjs";

const root = await mkdtemp(path.join(os.tmpdir(), "browsergrad-lock-test-"));

try {
  await testAcquireRelease();
  await testFreshLiveOwnerStays();
  await testDeadOwnerRemoved();
  await testReusedPidWithStaleHeartbeatRemoved();
  await testTokenReleaseDoesNotRemoveNewOwner();
  await testInvalidOwnerUsesDirectoryAge();
  console.log("cuda-lite tool lock tests passed");
} finally {
  await rm(root, { recursive: true, force: true });
}

async function testAcquireRelease() {
  const dir = path.join(root, "acquire-release.lock");
  const lock = await acquireDirectoryLock(dir, {
    heartbeatMs: 60_000,
    tool: "test",
  });
  const owner = await readLockOwner(dir);
  assert(owner?.token === lock.token, "owner token written");
  lock.stopHeartbeat();
  const released = await lock.release();
  assert(released, "owner released own lock");
  assert(await readLockOwner(dir) === undefined, "lock removed after release");
}

async function testFreshLiveOwnerStays() {
  const dir = path.join(root, "fresh-live.lock");
  await mkdir(dir);
  await writeLockOwner(dir, owner({
    pid: 123,
    token: "fresh-live",
    heartbeatAt: new Date(10_000).toISOString(),
  }));
  const removed = await removeStaleLock(dir, {
    now: () => 10_100,
    staleLockMs: 1_000,
    isProcessAlive: () => true,
  });
  assert(!removed, "fresh live owner preserved");
  assert((await readLockOwner(dir))?.token === "fresh-live", "fresh live owner still present");
}

async function testDeadOwnerRemoved() {
  const dir = path.join(root, "dead-owner.lock");
  await mkdir(dir);
  await writeLockOwner(dir, owner({
    pid: 456,
    token: "dead-owner",
    heartbeatAt: new Date(20_000).toISOString(),
  }));
  const removed = await removeStaleLock(dir, {
    now: () => 20_100,
    staleLockMs: 1_000,
    isProcessAlive: () => false,
  });
  assert(removed, "dead owner removed");
  assert(await readLockOwner(dir) === undefined, "dead owner lock gone");
}

async function testReusedPidWithStaleHeartbeatRemoved() {
  const dir = path.join(root, "reused-pid.lock");
  await mkdir(dir);
  await writeLockOwner(dir, owner({
    pid: process.pid,
    token: "reused-pid",
    heartbeatAt: new Date(30_000).toISOString(),
  }));
  const removed = await removeStaleLock(dir, {
    now: () => 40_000,
    staleLockMs: 1_000,
    isProcessAlive: () => true,
  });
  assert(removed, "stale heartbeat beats live pid");
  assert(await readLockOwner(dir) === undefined, "reused pid lock removed");
}

async function testTokenReleaseDoesNotRemoveNewOwner() {
  const dir = path.join(root, "token-safe.lock");
  const lock = await acquireDirectoryLock(dir, {
    heartbeatMs: 60_000,
    tool: "old",
  });
  await writeLockOwner(dir, owner({
    pid: process.pid,
    token: "new-owner",
    heartbeatAt: new Date(50_000).toISOString(),
  }));
  lock.stopHeartbeat();
  const released = await releaseDirectoryLock(dir, lock.token);
  assert(!released, "old token cannot release new owner");
  assert((await readLockOwner(dir))?.token === "new-owner", "new owner preserved");
  await rm(dir, { recursive: true, force: true });
}

async function testInvalidOwnerUsesDirectoryAge() {
  const dir = path.join(root, "invalid-owner.lock");
  await mkdir(dir);
  const removed = await removeStaleLock(dir, {
    now: () => Date.now() + 3_600_000,
    staleLockMs: 1,
    isProcessAlive: () => true,
  });
  assert(removed, "invalid owner removed after stale directory age");
  assert(await readLockOwner(dir) === undefined, "invalid owner lock gone");
}

function owner(overrides) {
  return {
    createdAt: new Date(0).toISOString(),
    heartbeatAt: new Date(0).toISOString(),
    pid: process.pid,
    token: "token",
    tool: "test",
    ...overrides,
  };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
