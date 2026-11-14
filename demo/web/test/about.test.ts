import { test } from "node:test";
import assert from "node:assert/strict";
import { HINTED, needs } from "../src/about.js";

test("a tab with a model it can use has no hint", () => {
  assert.equal(needs("embed", ["TASK_EMBED"]), "");
  assert.equal(needs("search", ["TASK_RERANK", "TASK_EMBED"]), "");
  assert.equal(needs("semantic", ["TASK_EXTRACTIVE_QA"]), "");
  assert.equal(needs("semantic", ["TASK_RERANK"]), "");
});

test("a tab with nothing to use says what it needs and where to get it", () => {
  assert.equal(needs("embed", []).startsWith("Needs: an embedding model."), true);
  assert.equal(needs("embed", ["TASK_RERANK"]).startsWith("Needs: an embedding model."), true);
  assert.equal(needs("search", ["TASK_RERANK"]).startsWith("Needs: an embedding model to create an index."), true);
  assert.equal(needs("semantic", []).startsWith("Needs: an embedding model, a reranker or a QA model."), true);
  assert.equal(needs("embed", []).includes("Server tab"), true);
});

test("the tabs that need no model never hint", () => {
  for (const tab of ["analyze", "vocabulary", "models", "server"]) {
    assert.equal(needs(tab, []), "");
    assert.equal(HINTED.includes(tab), false);
  }
});
