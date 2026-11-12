import { test } from "node:test";
import assert from "node:assert/strict";
import { blocks, cosineAt, extent, labelled, mix, nearest, share } from "../src/viz/matrix.js";

test("a cosine is read out of the upper triangle either way round", () => {
  // Four items: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
  const upper = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
  assert.equal(cosineAt(upper, 4, 0, 1), 0.1);
  assert.equal(cosineAt(upper, 4, 0, 3), 0.3);
  assert.equal(cosineAt(upper, 4, 1, 2), 0.4);
  assert.equal(cosineAt(upper, 4, 3, 1), 0.5);
  assert.equal(cosineAt(upper, 4, 2, 3), 0.6);
  assert.equal(cosineAt(upper, 4, 2, 2), 1);
  assert.equal(cosineAt([0.7], 2, 1, 0), 0.7);
});

test("shading is within the document's own range", () => {
  assert.deepEqual(extent([0.4, 0.9, 0.6]), [0.4, 0.9]);
  assert.deepEqual(extent([]), [0, 1]);
  assert.equal(share(0.4, 0.4, 0.9), 0);
  assert.equal(share(0.9, 0.4, 0.9), 1);
  assert.equal(share(1, 0.4, 0.9), 1);
  assert.equal(share(0.65, 0.4, 0.9), 0.5);
  assert.equal(share(0.3, 0.3, 0.3), 0.5);
});

test("a block spans the rows of the leaves it holds that have a row", () => {
  // Leaves 0..5; leaf 2 has no vector, so rows are 0,1,-,2,3,4.
  const rowOf = new Map([[0, 0], [1, 1], [3, 2], [4, 3], [5, 4]]);
  assert.deepEqual(blocks([[0, 1, 2], [3, 4, 5]], rowOf), [{ from: 0, to: 1 }, { from: 2, to: 4 }]);
  // One row left, or none: nothing to outline.
  assert.deepEqual(blocks([[1, 2], [2]], rowOf), []);
});

test("the nearest are the greatest cosines, the earlier of equals first", () => {
  assert.deepEqual(nearest([0.2, 0.9, undefined, 0.9, 0.5], 3), [{ at: 1, cosine: 0.9 }, { at: 3, cosine: 0.9 }, { at: 4, cosine: 0.5 }]);
  assert.deepEqual(nearest([0.2], 5), [{ at: 0, cosine: 0.2 }]);
  assert.deepEqual(nearest([undefined], 5), []);
});

test("axis labels thin out as the rows grow", () => {
  assert.deepEqual(labelled(5), [0, 1, 2, 3, 4]);
  assert.deepEqual(labelled(30), [0, 4, 9, 14, 19, 24, 29]);
  assert.equal(labelled(512).length, 11);
});

test("mixing runs from one end to the other and refuses nothing", () => {
  assert.equal(mix("#86b6ef", "#104281", 0), "rgb(134 182 239)");
  assert.equal(mix("#86b6ef", "#104281", 1), "rgb(16 66 129)");
  assert.equal(mix("#000000", "#ffffff", 0.5), "rgb(99 99 99)");
  assert.equal(mix("#86b6ef", "#104281", 7), "rgb(16 66 129)");
  assert.equal(mix("blue", "#104281", 0.5), "rgb(128 128 128)");
  assert.equal(mix("#zzzzzz", "#104281", 0.5), "rgb(128 128 128)");
});
