// What the Models tab works out from a catalog entry by itself: whether
// the artifact is for non-commercial use only, and whether bytes read
// from the server are the file the catalog lists. No DOM and no calls,
// so the unit tests run them under Node.

import type { CatalogFile } from "./gen/turbo_training_pb.js";

/** Whether an SPDX identifier is a non-commercial licence (CC BY-NC and its variants). */
function nc(licence: string): boolean {
  return /^CC-BY-NC(-|$)/.test(licence);
}

/** Whether the artifact may be used only non-commercially: its own licence, or data it was trained on. */
export function nonCommercial(e: { licence: string; trainedOn: readonly { licence: string }[] }): boolean {
  return nc(e.licence) || e.trainedOn.some((s) => nc(s.licence));
}

/** Why `got` bytes hashed to `sha256` are not the file the catalog lists, or "" when they are. */
export function mismatch(f: Pick<CatalogFile, "path" | "size" | "sha256">, got: number, sha256: string): string {
  if (BigInt(got) !== f.size) return `${f.path}: ${got} bytes arrived, the catalog lists ${f.size}`;
  if (sha256 !== f.sha256) return `${f.path}: the bytes have SHA-256 ${sha256}, the catalog lists ${f.sha256}`;
  return "";
}
