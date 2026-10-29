// Saving a model's file from the Models tab: the file's bytes through
// ReadModelFile (the same stream the library reads), hashed as they
// arrive and checked against the size and SHA-256 the catalog gives
// before the browser is handed them. There is no other download path.

import type { CatalogEntry, CatalogFile } from "../gen/turbo_training_pb.js";
import { catalog } from "../rpc.js";
import { Sha256 } from "../sha256.js";
import { size } from "../dom.js";
import { mismatch } from "../modelfile.js";

export { nonCommercial } from "../modelfile.js";

/** Reads `f` of `e`, checks it, and saves it under its own name. */
export async function download(e: CatalogEntry, f: CatalogFile, status: HTMLElement, signal: AbortSignal): Promise<void> {
  const parts: Uint8Array<ArrayBuffer>[] = [];
  const hash = new Sha256();
  let got = 0;
  for await (const m of catalog.readModelFile({ artifactId: e.id, path: f.path, sha256: f.sha256 }, { signal })) {
    if (m.frame.case !== "data") continue;
    const bytes = new Uint8Array(m.frame.value);
    parts.push(bytes);
    hash.update(bytes);
    got += bytes.length;
    status.textContent = `${e.id}/${f.path}: ${size(got)} of ${size(f.size)}`;
  }
  const wrong = mismatch(f, got, hash.hex());
  if (wrong) throw new Error(wrong);
  const url = URL.createObjectURL(new Blob(parts, { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = f.path.split("/").pop() ?? f.path;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  status.textContent = `${e.id}/${f.path}: ${size(f.size)}, SHA-256 checked, saved.`;
}
