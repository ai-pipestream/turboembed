// What the server serves, read once per refresh and shared by the tabs,
// and whether the server answers at all.

import type { CapabilitiesResponse } from "./gen/turbo_analysis_pb.js";
import * as permit from "./permit.js";
import { analysis, demo } from "./rpc.js";
import { stale, type Status } from "./status.js";

let current: Promise<CapabilitiesResponse> | null = null;
const listeners: ((c: CapabilitiesResponse) => void)[] = [];
const failures: ((e: unknown) => void)[] = [];

/** The server's capabilities, fetched on first use and after `refresh`. */
export function capabilities(): Promise<CapabilitiesResponse> {
  if (!current) {
    current = analysis.capabilities({});
    current.then(
      (c) => listeners.forEach((l) => l(c)),
      (e) => failures.forEach((l) => l(e)),
    );
  }
  return current;
}

export function refresh(): Promise<CapabilitiesResponse> {
  current = null;
  return capabilities();
}

/** Calls `l` with each new capabilities answer. */
export function onCapabilities(l: (c: CapabilitiesResponse) => void): void {
  listeners.push(l);
}

/** Calls `l` with the error each time the capabilities could not be read. */
export function onCapabilitiesFailed(l: (e: unknown) => void): void {
  failures.push(l);
}

/** The served names of the bundles with `task` (TASK_EMBED, TASK_RERANK, ...). */
export function bundles(c: CapabilitiesResponse, task: string): string[] {
  return c.bundles.filter((b) => b.task === task).map((b) => b.name);
}

// The last status check's outcome: unknown until the first check answers
// or fails.
let last: Status | null = null;
const watchers: ((up: boolean, was: boolean | null) => void)[] = [];
const readers: (() => void)[] = [];

/** Whether the server answered the last status check; null before the first. */
export function reachable(): boolean | null {
  return last === null ? null : last.up;
}

/**
 * Records a status check's outcome: whether the server answered, and
 * whether it said ready. Tells the watchers when reachability changed,
 * and the readers when their lists are worth reading again.
 */
export function setStatus(up: boolean, ready: boolean): void {
  const was = last;
  last = { up, ready: up && ready };
  if (was === null || was.up !== up) watchers.forEach((l) => l(up, was === null ? null : was.up));
  if (stale(was, last)) readers.forEach((l) => l());
}

/** Calls `l` when the server stops or starts answering, with what it was before. */
export function onReachable(l: (up: boolean, was: boolean | null) => void): void {
  watchers.push(l);
}

/**
 * Calls `l` whenever a list read from the server is worth reading:
 * once the first status check has an outcome (so `reachable()` is known
 * by the time the read fails or answers), when the server stops
 * answering (so the list says so), when it answers again, and when it
 * turns ready after loading (status.ts). A tab reads its lists here and
 * nowhere else at start.
 */
export function onServed(l: () => void): void {
  readers.push(l);
  if (last !== null) l();
}

/** Why a list of served things is empty: no server, or a server that serves none. */
export function whyNone(what: string): string {
  return reachable() === false ? "the server is not reachable" : `no ${what} is served`;
}

/** Asks the demo which writes it forwards; with no answer, nothing is turned off. */
export async function permits(): Promise<void> {
  try {
    const d = await demo.describe({});
    permit.set(d.forwardedWrites, d.hasUser);
  } catch {
    permit.set(null);
  }
}
