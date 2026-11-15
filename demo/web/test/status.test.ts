import { test } from "node:test";
import assert from "node:assert/strict";
import { stale } from "../src/status.js";

const s = (up: boolean, ready: boolean) => ({ up, ready });

test("the first outcome reads the lists, whatever it is", () => {
  assert.equal(stale(null, s(true, true)), true);
  assert.equal(stale(null, s(true, false)), true);
  assert.equal(stale(null, s(false, false)), true);
});

test("going down and coming back both read the lists again", () => {
  assert.equal(stale(s(true, true), s(false, false)), true);
  assert.equal(stale(s(false, false), s(true, true)), true);
  assert.equal(stale(s(false, false), s(true, false)), true);
});

test("turning ready reads the lists again, and nothing else does", () => {
  assert.equal(stale(s(true, false), s(true, true)), true);
  assert.equal(stale(s(true, true), s(true, true)), false);
  assert.equal(stale(s(true, false), s(true, false)), false);
  assert.equal(stale(s(true, true), s(true, false)), false);
  assert.equal(stale(s(false, false), s(false, false)), false);
});
