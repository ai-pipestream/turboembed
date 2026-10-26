// What the server serves, read once per refresh and shared by the tabs.

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
