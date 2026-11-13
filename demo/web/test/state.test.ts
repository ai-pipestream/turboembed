// Late answers from earlier capabilities or permits calls must not
// overwrite the answer of a newer call.

import assert from "node:assert/strict";
import { test } from "node:test";
import { create } from "@bufbuild/protobuf";
import { CapabilitiesResponseSchema, type CapabilitiesResponse } from "../src/gen/turbo_analysis_pb.js";
import { off, set } from "../src/permit.js";
import { capabilities, onCapabilities, onCapabilitiesFailed, permits, refresh } from "../src/state.js";

function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void } {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function caps(names: string[]): CapabilitiesResponse {
  return create(CapabilitiesResponseSchema, {
    bundles: names.map((name) => ({ name, task: "TASK_EMBED" })),
  });
}

test("a late capabilities answer from an older refresh does not overwrite a newer one", async () => {
  const seen: string[][] = [];
  const errors: string[] = [];
  onCapabilities((c) => seen.push(c.bundles.map((b) => b.name)));
  onCapabilitiesFailed((e) => errors.push(e instanceof Error ? e.message : String(e)));

  const older = deferred<CapabilitiesResponse>();
  const newer = deferred<CapabilitiesResponse>();

  const p1 = refresh(() => older.promise);
  // A concurrent capabilities() call before refresh() shares the in-flight call.
  const p2 = refresh(() => newer.promise);
  assert.equal(capabilities(() => older.promise), p2);

  newer.resolve(caps(["newer-bundle"]));
  await p2;
  assert.deepEqual(seen, [["newer-bundle"]]);

  older.resolve(caps(["older-bundle"]));
  await p1;
  assert.deepEqual(seen, [["newer-bundle"]]);
  assert.deepEqual(errors, []);
});

test("a late capabilities failure from an older refresh does not overwrite a newer answer", async () => {
  const seen: string[][] = [];
  const errors: string[] = [];
  onCapabilities((c) => seen.push(c.bundles.map((b) => b.name)));
  onCapabilitiesFailed((e) => errors.push(e instanceof Error ? e.message : String(e)));

  const older = deferred<CapabilitiesResponse>();
  const newer = deferred<CapabilitiesResponse>();

  const p1 = refresh(() => older.promise);
  const p2 = refresh(() => newer.promise);

  newer.resolve(caps(["ready-bundle"]));
  await p2;
  assert.deepEqual(seen, [["ready-bundle"]]);

  older.reject(new Error("stale failure"));
  await p1.catch(() => undefined);
  assert.deepEqual(seen, [["ready-bundle"]]);
  assert.deepEqual(errors, []);
});

test("a late capabilities success from an older refresh does not hide a newer failure", async () => {
  const seen: string[][] = [];
  const errors: string[] = [];
  onCapabilities((c) => seen.push(c.bundles.map((b) => b.name)));
  onCapabilitiesFailed((e) => errors.push(e instanceof Error ? e.message : String(e)));

  const older = deferred<CapabilitiesResponse>();
  const newer = deferred<CapabilitiesResponse>();

  const p1 = refresh(() => older.promise);
  const p2 = refresh(() => newer.promise);

  newer.reject(new Error("latest failed"));
  await p2.catch(() => undefined);
  assert.deepEqual(errors, ["latest failed"]);

  older.resolve(caps(["stale-bundle"]));
  await p1;
  assert.deepEqual(seen, []);
  assert.deepEqual(errors, ["latest failed"]);
});

test("a late permits answer from an older call does not overwrite a newer one", async () => {
  const createIndex = { name: "CreateIndex", parent: { typeName: "turbo.search.v1.TurboSearchService" } };
  const deleteIndex = { name: "DeleteIndex", parent: { typeName: "turbo.search.v1.TurboSearchService" } };
  set(null);

  const older = deferred<{ forwardedWrites: string[]; hasUser: boolean }>();
  const newer = deferred<{ forwardedWrites: string[]; hasUser: boolean }>();

  const p1 = permits(() => older.promise);
  const p2 = permits(() => newer.promise);

  newer.resolve({ forwardedWrites: ["/turbo.search.v1.TurboSearchService/CreateIndex"], hasUser: true });
  await p2;
  assert.equal(off(createIndex), "");
  assert.equal(off(deleteIndex) !== "", true);

  older.resolve({ forwardedWrites: [], hasUser: false });
  await p1;
  assert.equal(off(createIndex), "");
  assert.equal(off(deleteIndex) !== "", true);
  set(null);
});
