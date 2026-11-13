// Errors for a person: a plain sentence first, the protocol's words as
// details.

import assert from "node:assert/strict";
import { test } from "node:test";
import { Code, ConnectError } from "@connectrpc/connect";
import { cancelled, codeName, explain, sentence, unprefixed } from "../src/errors.js";

function refusal(message: string, code: Code, meta: Record<string, string> = {}): ConnectError {
  return new ConnectError(message, code, new Headers(meta));
}

test("an empty index name says what to do, and keeps the protocol's words as details", () => {
  const x = explain(refusal("TURBO_E_INVALID_ARGUMENT: index_name must not be empty", Code.InvalidArgument, { "turbo-code": "256", "turbo-field": "1" }));
  assert.equal(x.text, "Pick or create an index first.");
  assert.equal(x.details, "invalid_argument, turbo-code 256, field 1: TURBO_E_INVALID_ARGUMENT: index_name must not be empty");
});

test("an unknown refusal leads with the server's message as a sentence", () => {
  const x = explain(refusal("TURBO_E_INVALID_ARGUMENT: top_k is over the limit of 1000", Code.InvalidArgument, { "turbo-code": "256" }));
  assert.equal(x.text, "Top_k is over the limit of 1000.");
  assert.equal(x.details.startsWith("invalid_argument, turbo-code 256: "), true);
});

test("a write off the allowlist names the flag that enables it", () => {
  const x = explain(refusal("/turbo.search.v1.TurboSearchService/DeleteIndex is not on the demo's allowlist", Code.PermissionDenied));
  assert.equal(x.text, "The demo does not forward this write. Start it with --allow turbo.search.v1.TurboSearchService/DeleteIndex");
  assert.equal(x.details, "permission_denied: /turbo.search.v1.TurboSearchService/DeleteIndex is not on the demo's allowlist");
});

test("a missing token, a server that is away and a slow call each read plainly", () => {
  assert.equal(explain(refusal("a bearer token is required", Code.Unauthenticated)).text, "The demo wants its bearer token. Enter it at the top of the page.");
  assert.equal(explain(refusal("the server: tcp connect error", Code.Unavailable)).text, "The server is not reachable through the demo.");
  assert.equal(explain(refusal("", Code.DeadlineExceeded)).text, "The call took too long and was stopped.");
  assert.equal(explain(refusal("", Code.Internal)).text, "The call failed (internal).");
});

test("an error the page raised is its own message, with no details", () => {
  assert.deepEqual(explain(new Error("Enter at least one open-span label.")), { text: "Enter at least one open-span label.", details: "" });
});

test("a stopped call is cancelled, not a failure", () => {
  const abort = new Error("stop");
  abort.name = "AbortError";
  assert.equal(cancelled(abort), true);
  assert.equal(cancelled(refusal("", Code.Canceled)), true);
  assert.equal(cancelled(refusal("", Code.Unavailable)), false);
  assert.equal(explain(abort).text, "Cancelled.");
});

test("only a TURBO_E_ prefix is taken off a message", () => {
  assert.equal(unprefixed("TURBO_E_NOT_FOUND: no index demo"), "no index demo");
  assert.equal(unprefixed("note: TURBO_E_X: y"), "note: TURBO_E_X: y");
  assert.equal(unprefixed("TURBO_E_ bad: y"), "TURBO_E_ bad: y");
});

test("a sentence starts with a capital, whatever the script, and ends once", () => {
  assert.equal(sentence("no index demo"), "No index demo.");
  assert.equal(sentence("élan vital"), "Élan vital.");
  assert.equal(sentence("Done."), "Done.");
  assert.equal(sentence("  "), "");
  assert.equal(codeName(Code.PermissionDenied), "permission_denied");
});
