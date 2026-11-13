// Only the latest call of a kind may draw its answer.

import assert from "node:assert/strict";
import { test } from "node:test";
import { latest } from "../src/latest.js";

test("a ticket holds until a later call takes one", () => {
  const fresh = latest();
  const first = fresh();
  assert.equal(first(), true);
  const second = fresh();
  assert.equal(first(), false);
  assert.equal(second(), true);
});

test("two kinds of call do not end each other", () => {
  const [a, b] = [latest(), latest()];
  const one = a();
  b();
  b();
  assert.equal(one(), true);
});

test("an older answer that arrives last is not drawn", async () => {
  const fresh = latest();
  const drawn: string[] = [];
  const call = async (name: string, wait: Promise<void>): Promise<void> => {
    const mine = fresh();
    await wait;
    if (!mine()) return;
    drawn.push(name);
  };
  let release = (): void => {};
  const slow = new Promise<void>((r) => (release = r));
  const older = call("older", slow);
  await call("newer", Promise.resolve());
  release();
  await older;
  assert.deepEqual(drawn, ["newer"]);
});
