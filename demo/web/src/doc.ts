// The document the Analyze tab holds: its binary DocumentGraph bytes and
// views, shared by the panels that inspect it (window, X-Ray, annotation
// drawer, projection, wire inspector).

import type { ViewInfo } from "./gen/turbo_typed_annotations_pb.js";

export type Doc = { graph: Uint8Array; views: ViewInfo[]; source: string };

let current: Doc | null = null;
const listeners: ((d: Doc | null) => void)[] = [];

/** The document now held, or null before the first analysis. */
export function doc(): Doc | null {
  return current;
}

/** Holds `d` and tells every panel; null clears it. */
export function setDoc(d: Doc | null): void {
  current = d;
  for (const l of listeners) l(d);
}

export function onDoc(l: (d: Doc | null) => void): void {
  listeners.push(l);
}
