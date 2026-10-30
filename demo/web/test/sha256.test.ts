// The page's SHA-256 against Node's, across block edges and in pieces.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { Sha256, sha256Blob } from "../src/sha256.js";

function bytes(n: number, seed: number): Uint8Array {
  const b = new Uint8Array(n);
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    b[i] = x % 256;
  }
  return b;
}

test("every length from 0 to 300 bytes", () => {
  for (let n = 0; n <= 300; n++) {
    const b = bytes(n, n + 1);
    const h = new Sha256();
    h.update(b);
    assert.equal(h.hex(), createHash("sha256").update(b).digest("hex"), `length ${n}`);
  }
});

test("updates in odd pieces give the one-shot digest", () => {
  const b = bytes(10_000, 7);
  const h = new Sha256();
  for (let at = 0, step = 1; at < b.length; at += step, step = (step * 3) % 97 + 1) h.update(b.subarray(at, Math.min(b.length, at + step)));
  assert.equal(h.hex(), createHash("sha256").update(b).digest("hex"));
});

test("a Blob streamed in chunks", async () => {
  const b = bytes(3 * 1024 * 1024 + 5, 11);
  let seen = 0;
  const hex = await sha256Blob(new Blob([b as Uint8Array<ArrayBuffer>]), (n) => (seen = n));
  assert.equal(hex, createHash("sha256").update(b).digest("hex"));
  assert.equal(seen, b.length);
});
