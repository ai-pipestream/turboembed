// Which writes are off: none until the demo answers, then each one the
// demo does not forward, with the flag that enables it.

import assert from "node:assert/strict";
import { test } from "node:test";
import { off, path, set } from "../src/permit.js";

const create = { name: "CreateIndex", parent: { typeName: "turbo.search.v1.TurboSearchService" } };
const remove = { name: "DeleteIndex", parent: { typeName: "turbo.search.v1.TurboSearchService" } };

test("a method's path is its service and name", () => {
  assert.equal(path(create), "/turbo.search.v1.TurboSearchService/CreateIndex");
});

test("nothing is off before the demo answers, or when it gave no answer", () => {
  assert.equal(off(create), "");
  set(null);
  assert.equal(off(create), "");
});

test("a write the demo does not forward is off, and says which --allow enables it", () => {
  set([path(create)]);
  assert.equal(off(create), "");
  assert.equal(off(remove), "Off: the demo does not forward this write. Start it with --allow turbo.search.v1.TurboSearchService/DeleteIndex");
  set([]);
  assert.equal(off(create).endsWith("--allow turbo.search.v1.TurboSearchService/CreateIndex"), true);
});

test("a catalog write also needs the demo's user", () => {
  const accept = { name: "RecordLicenceAcceptance", parent: { typeName: "turbo.training.v1.TurboCatalogService" } };
  set([path(accept), path(create)], false);
  assert.equal(off(accept), "Off: this acts for a user. Start the demo with --user NAME");
  assert.equal(off(create), "");
  set([path(accept)], true);
  assert.equal(off(accept), "");
});
