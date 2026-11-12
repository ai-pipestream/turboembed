// What the server serves, read once per refresh and shared by the tabs,
// and whether the server answers at all.

import type { CapabilitiesResponse } from "./gen/turbo_analysis_pb.js";
import { analysis } from "./rpc.js";

let current: Promise<CapabilitiesResponse> | null = null;
const listeners: ((c: CapabilitiesResponse) => void)[] = [];

/** The server's capabilities, fetched on first use and after `refresh`. */
export function capabilities(): Promise<CapabilitiesResponse> {
  if (!current) {
    current = analysis.capabilities({});
    current.then((c) => listeners.forEach((l) => l(c))).catch(() => undefined);
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

/** The served names of the bundles with `task` (TASK_EMBED, TASK_RERANK, ...). */
export function bundles(c: CapabilitiesResponse, task: string): string[] {
  return c.bundles.filter((b) => b.task === task).map((b) => b.name);
}

// Whether the last status check reached the server: unknown until the
// first check answers.
let up: boolean | null = null;
const watchers: ((up: boolean, was: boolean | null) => void)[] = [];

/** Whether the server answered the last status check; null before the first. */
export function reachable(): boolean | null {
  return up;
}

/** Records a status check's outcome and tells the watchers when it changed. */
export function setReachable(now: boolean): void {
  if (now === up) return;
  const was = up;
  up = now;
  watchers.forEach((l) => l(now, was));
}

/** Calls `l` when the server stops or starts answering, with what it was before. */
export function onReachable(l: (up: boolean, was: boolean | null) => void): void {
  watchers.push(l);
}

/** Calls `l` each time the server answers again after it did not. */
export function onBack(l: () => void): void {
  watchers.push((now, was) => {
    if (now && was === false) l();
  });
}

/** Why a list of served things is empty: no server, or a server that serves none. */
export function whyNone(what: string): string {
  return up === false ? "the server is not reachable" : `no ${what} is served`;
}
