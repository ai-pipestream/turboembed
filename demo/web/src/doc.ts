// The document the Analyze tab holds: its binary DocumentGraph bytes and
// views, shared by the panels that inspect it (window, X-Ray, annotation
// drawer, projection, wire inspector).

import type { ViewInfo } from "./gen/turbo_typed_annotations_pb.js";

/** `vectors`: whether the graph holds chunk or sentence vectors, so a panel knows before it asks. */
export type Doc = { graph: Uint8Array; views: ViewInfo[]; source: string; vectors: boolean };

/** The layers ProjectChunks places and compares: vectors over the chunks, else over the sentences. */
const PLACED = ["turbo:embeddings<turbo:chunks>", "turbo:embeddings<turbo:sentences>"];

/** Whether a graph's layer list names vectors a projection can use. */
export function hasVectors(layers: { layer: string }[]): boolean {
  return layers.some((l) => PLACED.includes(l.layer));
}

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
