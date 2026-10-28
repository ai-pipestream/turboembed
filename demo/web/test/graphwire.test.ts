// The DocumentGraph reader walks the analysis crate's checked-in graph
// fixture end to end, names where every byte goes, and refuses bytes
// that are cut short, padded or not a graph.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { WireError, walkGraph, type Part } from "../src/graphwire.js";

// build-web.sh runs the tests from the repository root.
const FIXTURE = new Uint8Array(readFileSync("analysis/core/tests/fixtures/document-graph.bin"));

function find(p: Part, name: string): Part {
  const f = p.parts?.find((x) => x.name === name);
  if (!f) throw new Error(`no part ${name} in ${p.name}`);
  return f;
}

function nested(p: Part): void {
  let at = p.at;
  for (const c of p.parts ?? []) {
    assert.equal(c.at >= at, true, `${c.name} starts inside the part before it`);
    assert.equal(c.at + c.len <= p.at + p.len, true, `${c.name} ends past ${p.name}`);
    at = c.at + c.len;
    nested(c);
  }
}

function all(p: Part): Part[] {
  return [p, ...(p.parts ?? []).flatMap(all)];
}

test("the fixture reads to its end, every part inside its parent", () => {
  const g = walkGraph(FIXTURE);
  assert.equal(g.len, FIXTURE.length);
  let at = 0;
  for (const p of g.parts ?? []) {
    assert.equal(p.at, at, `${p.name} follows the part before it`);
    at += p.len;
  }
  assert.equal(at, FIXTURE.length);
  nested(g);
  assert.equal(find(g, "magic").value, "TGRF");
  assert.equal(find(g, "format version").value, "1");
});

test("the fixture's views, alignment and statuses are where the export put them", () => {
  const g = walkGraph(FIXTURE);
  const align = find(g, "source alignment");
  assert.equal(find(align, "edits").value, "1 edits");
  const views = find(g, "views");
  // The default view's instance is the empty name.
  assert.deepEqual(views.parts?.map((v) => v.value), ["", "clean"]);
  const statuses = all(find(g, "layers")).filter((p) => p.name === "status").map((p) => p.value ?? "");
  assert.equal(statuses.some((s) => s.startsWith("skipped")), true, "a skipped layer");
  assert.equal(statuses.includes("failed"), true, "a failed layer");
  const models = all(g).filter((p) => p.name === "id").map((p) => p.value);
  assert.equal(models.includes("test/model"), true);
});

test("cut, padded or foreign bytes are refused with where", () => {
  for (const cut of [3, 6, 20, FIXTURE.length - 1]) {
    assert.throws(() => walkGraph(FIXTURE.subarray(0, cut)), WireError);
  }
  const padded = new Uint8Array(FIXTURE.length + 1);
  padded.set(FIXTURE);
  assert.throws(() => walkGraph(padded), (e: unknown) => e instanceof WireError && e.at === FIXTURE.length);
  const foreign = FIXTURE.slice();
  foreign[0] = 0x58;
  assert.throws(() => walkGraph(foreign), (e: unknown) => e instanceof WireError && e.at === 0);
});
