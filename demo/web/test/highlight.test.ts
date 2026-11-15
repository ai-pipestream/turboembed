// A hit's matched spans marked in its text.

import assert from "node:assert/strict";
import { test } from "node:test";
import { pieces, unplaced, type Match } from "../src/highlight.js";

const m = (source: string, label: string, start: number, end: number): Match => ({ source, label, start, end });

test("a match is cut out of the text at its byte offsets, from the chunk's start", () => {
  // The chunk starts at byte 100 of its document.
  const got = pieces("Rust is a memory safe language.", 100, [m("bm25_term", "memory", 110, 116), m("bm25_term", "safe", 117, 121)]);
  assert.deepEqual(got.map((p) => [p.text, p.sources.join()]), [
    ["Rust is a ", ""],
    ["memory", "bm25_term"],
    [" ", ""],
    ["safe", "bm25_term"],
    [" language.", ""],
  ]);
  assert.deepEqual(got[1].labels, ["memory"]);
});

test("overlapping matches from two signals share the overlap", () => {
  const got = pieces("memory safe!", 0, [m("bm25_term", "memory", 0, 6), m("phrase", "memory safe", 0, 11)]);
  assert.deepEqual(got.map((p) => [p.text, p.sources]), [
    ["memory", ["bm25_term", "phrase"]],
    [" safe", ["phrase"]],
    ["!", []],
  ]);
});

test("offsets are bytes, and a cut never splits a character", () => {
  // "é" is two bytes: "café" ends at byte 5, and an end at byte 4 is inside it.
  assert.deepEqual(pieces("café au lait", 0, [m("bm25_term", "au", 6, 8)]).map((p) => p.text), ["café ", "au", " lait"]);
  assert.deepEqual(pieces("café au", 0, [m("bm25_term", "caf", 0, 4)]).map((p) => p.text), ["caf", "é au"]);
});

test("a match over the whole text or outside it marks nothing and is reported apart", () => {
  const whole = m("dense", "", 100, 104);
  const outside = m("bm25_term", "x", 10, 12);
  assert.deepEqual(pieces("Rust", 100, [whole, outside]), [{ text: "Rust", sources: [], labels: [] }]);
  assert.deepEqual(unplaced("Rust", 100, [whole, outside, m("bm25_term", "us", 101, 103)]), [whole, outside]);
});

test("an empty text has no pieces", () => {
  assert.deepEqual(pieces("", 0, []), []);
});
