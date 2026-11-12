import assert from "node:assert/strict";
import { test } from "node:test";
import { count, elapsed, size } from "../src/dom.js";

test("elapsed time reads in tenths of a second under a minute", () => {
  assert.equal(elapsed(0), "0.0 s");
  assert.equal(elapsed(1234), "1.2 s");
  assert.equal(elapsed(59_949), "59.9 s");
});

test("elapsed time reads in minutes and whole seconds from a minute on", () => {
  assert.equal(elapsed(60_000), "1 min 00 s");
  assert.equal(elapsed(125_900), "2 min 05 s");
});

test("sizes read in binary units", () => {
  assert.equal(size(512), "512 B");
  assert.equal(size(1536), "1.5 KiB");
  assert.equal(size(3 * 1024 * 1024), "3.0 MiB");
});

test("a count agrees with its noun", () => {
  assert.equal(count(1, "document"), "1 document");
  assert.equal(count(0, "document"), "0 documents");
  assert.equal(count(12, "document"), "12 documents");
});
