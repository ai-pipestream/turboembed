// The stop wait in stack.ts, on real processes: one that exits on
// SIGTERM ends the wait, and one that ignores it is killed and named
// instead of hanging the run.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { test } from "node:test";
import { ended } from "./stack.js";

/** A shell that waits forever, ignoring SIGTERM when `stubborn`. */
function waiting(stubborn: boolean) {
  const trap = stubborn ? `trap "" TERM; ` : "";
  return spawn("sh", ["-c", `${trap}while :; do sleep 0.1; done`], { stdio: ["ignore", "pipe", "pipe"] });
}

test("a process that exits on SIGTERM ends the wait", async () => {
  const child = waiting(false);
  await ended(child, "polite", 5_000);
  assert.equal(child.signalCode, "SIGTERM");
});

test("a process that ignores SIGTERM is killed and named", async () => {
  const child = waiting(true);
  // Let the shell set its trap before it is signalled.
  await new Promise((resolve) => setTimeout(resolve, 200));
  const started = Date.now();
  await assert.rejects(ended(child, "stubborn", 500), /^Error: stubborn did not stop within 500 ms of SIGTERM and was killed\./);
  assert.equal(child.signalCode, "SIGKILL");
  assert.equal(Date.now() - started < 5_000, true);
});
