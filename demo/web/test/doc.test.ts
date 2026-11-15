// Whether a graph's layer list holds vectors the Document map can ask a
// matrix of.

import assert from "node:assert/strict";
import { test } from "node:test";
import { hasVectors } from "../src/doc.js";
import { LayerWireStatus } from "../src/gen/turbo_typed_annotations_pb.js";

const CHUNKS = "turbo:embeddings<turbo:chunks>";
const layer = (layer: string, status = LayerWireStatus.OK, items = 4, instance = "") => ({ layer, instance, status, items });

test("chunk or sentence vectors that ran and hold items count", () => {
  assert.equal(hasVectors([layer("turbo:tokens"), layer(CHUNKS)]), true);
  assert.equal(hasVectors([layer("turbo:embeddings<turbo:sentences>")]), true);
  assert.equal(hasVectors([layer("turbo:tokens"), layer("turbo:chunks")]), false);
  assert.equal(hasVectors([]), false);
});

test("a vectors layer that was skipped, failed, is empty or is a named instance does not count", () => {
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.SKIPPED, 0)]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.FAILED, 0)]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.OK, 0)]), false);
  // The status decides on its own, whatever count a layer that did not run reports.
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.SKIPPED, 4)]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.FAILED, 4)]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.UNSPECIFIED, 4)]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.OK, 4, "other")]), false);
  assert.equal(hasVectors([layer(CHUNKS, LayerWireStatus.OK, 4, "other"), layer(CHUNKS)]), true);
});
