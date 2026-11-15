import { test } from "node:test";
import assert from "node:assert/strict";
import { aroundGap, cutReason, distinctive, hull, leaves, pieces, regions, riverTerms, scale, spread, streams, type Unit } from "../src/viz/model.js";

const u = (item: number, start: number, end: number, level: number, x = 0, y = 0, empty = false): Unit => ({ item, start, end, level, x, y, z: 0, empty });

test("the finest level gives the points, in reading order", () => {
  const all = [u(0, 0, 100, 2), u(3, 60, 100, 0), u(1, 0, 30, 0), u(4, 0, 60, 1), u(2, 30, 60, 0)];
  assert.deepEqual(leaves(all).map((x) => x.item), [1, 2, 3]);
  assert.deepEqual(leaves([]), []);
});

test("a coarser unit is a region only when it holds some of the points, not one and not all", () => {
  const all = [u(0, 0, 100, 2), u(1, 0, 30, 0), u(2, 30, 60, 0), u(3, 60, 100, 0), u(4, 0, 60, 1), u(5, 60, 100, 1)];
  const r = regions(all, leaves(all));
  assert.deepEqual(r.map((x) => [x.unit.item, x.members]), [[4, [0, 1]]]);
});

test("a convex hull drops the inner points and keeps a line or a point as it is", () => {
  const h = hull([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]]);
  assert.equal(h.length, 4);
  assert.equal(h.some(([x, y]) => x === 1 && y === 1), false);
  assert.deepEqual(hull([[1, 1], [1, 1]]), [[1, 1]]);
  assert.deepEqual(hull([[0, 0], [3, 3]]), [[0, 0], [3, 3]]);
});

test("a scale maps its ends and puts a flat domain in the middle", () => {
  const s = scale(-1, 1, 0, 200);
  assert.equal(s(-1), 0);
  assert.equal(s(0), 100);
  assert.equal(scale(5, 5, 0, 10)(5), 5);
});

test("the words every chunk shares are not what tells one apart, in any script", () => {
  const doc = [{ term: "der", weight: 10 }, { term: "über", weight: 4 }, { term: "世界", weight: 3 }, { term: "bank", weight: 3 }];
  const chunk = [{ term: "der", weight: 2 }, { term: "世界", weight: 3 }, { term: "zins", weight: 1 }];
  const got = distinctive(chunk, 6, doc, 20, 2);
  assert.deepEqual(got, ["世界", "zins"]);
  assert.deepEqual(distinctive([], 0, doc, 20, 3), []);
});

test("a word most chunks list is not what tells one apart", () => {
  const doc = [{ term: "the", weight: 10 }, { term: "engine", weight: 3 }, { term: "rates", weight: 3 }];
  const c1 = [{ term: "the", weight: 6 }, { term: "engine", weight: 3 }];
  const c2 = [{ term: "the", weight: 2 }, { term: "rates", weight: 3 }];
  const c3 = [{ term: "the", weight: 2 }];
  const listed = spread([c1, c2, c3]);
  assert.equal(listed.get("the"), 3);
  assert.deepEqual(distinctive(c1, 10, doc, 30, 2, listed, 3), ["engine"]);
});

test("the river takes the terms that tell chunks apart, in the order of their chunks, or the heaviest when asked", () => {
  const doc = [{ term: "the", weight: 10 }, { term: "rates", weight: 3 }, { term: "engine", weight: 4 }];
  const chunks = [
    { terms: [{ term: "the", weight: 5 }, { term: "engine", weight: 4 }], total: 10 },
    { terms: [{ term: "the", weight: 5 }, { term: "rates", weight: 3 }], total: 10 },
  ];
  assert.deepEqual(riverTerms(doc, 20, chunks, 2, true), ["engine", "rates"]);
  assert.deepEqual(riverTerms(doc, 20, chunks, 2, false), ["the", "engine"]);
});

test("streams stack about the centre line and touch", () => {
  const s = streams([[2, 0], [4, 2]]);
  assert.deepEqual(s[0].lo, [-3, -1]);
  assert.deepEqual(s[0].hi, [-1, -1]);
  assert.deepEqual(s[1].lo, [-1, -1]);
  assert.deepEqual(s[1].hi, [3, 1]);
  assert.deepEqual(streams([]), []);
});

test("the text is cut at chunk and mark edges, and a mark may cross a chunk edge", () => {
  const text = "Ada wrote. Grüße aus Köln!";
  const got = pieces(text, [{ start: 0, end: 10 }, { start: 11, end: text.length }], [[4, 16]]);
  assert.deepEqual(got.map((p) => [p.text, p.chunk, p.marked]), [
    ["Ada ", 0, false],
    ["wrote.", 0, true],
    [" ", null, true],
    ["Grüße", 1, true],
    [" aus Köln!", 1, false],
  ]);
  assert.equal(got.map((p) => p.text).join(""), text);
});

test("a gap shows the sentence on each side", () => {
  const text = "One. Two. Three.";
  assert.deepEqual(aroundGap(text, [4, 9, 16], 1), ["Two.", "Three."]);
  assert.deepEqual(aroundGap(text, [4, 9, 16], 0), ["One.", "Two."]);
});

test("a cut reason reads as words, and an unknown code is still shown", () => {
  assert.equal(cutReason(4), "paragraph");
  assert.equal(cutReason(5), "topic shift");
  assert.equal(cutReason(42), "reason 42");
});
