// Similarity colours follow the matrix's own range.

import assert from "node:assert/strict";
import { test } from "node:test";
import { best, place, range, strength } from "../src/heat.js";

const narrow = [
  [1, 0.97, 0.89],
  [0.97, 1, 0.93],
  [0.89, 0.93, 1],
];

test("the range is taken off the diagonal", () => {
  assert.deepEqual(range(narrow), [0.89, 0.97]);
  assert.deepEqual(range([[1]]), [0, 1]);
  assert.deepEqual(range([]), [0, 1]);
});

test("values in a narrow band spread over the whole scale", () => {
  const [lo, hi] = range(narrow);
  assert.equal(place(0.89, lo, hi), 0);
  assert.equal(place(0.97, lo, hi), 1);
  assert.equal(Math.abs(place(0.93, lo, hi) - 0.5) < 1e-9, true);
  // On the absolute scale the same three values are within 8% of each other.
  assert.equal(place(0.97, 0, 1) - place(0.89, 0, 1) < 0.09, true);
  assert.equal(strength(place(0.97, lo, hi)) - strength(place(0.89, lo, hi)) > 0.7, true);
});

test("the diagonal clamps to the top and an empty range is the top", () => {
  assert.equal(place(1, 0.89, 0.97), 1);
  assert.equal(place(-0.2, 0, 1), 0);
  assert.equal(place(0.5, 0.5, 0.5), 1);
});

test("each row's best match is its greatest value off the diagonal", () => {
  assert.equal(best(narrow, 0), 1);
  assert.equal(best(narrow, 2), 1);
  assert.equal(best([[1]], 0), -1);
  assert.equal(best([[1, NaN], [NaN, 1]], 0), -1);
});
