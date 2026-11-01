// The Models tab's words for a model source, and the redirect back from
// Google's consent page.

import { test } from "node:test";
import assert from "node:assert/strict";
import { create } from "@bufbuild/protobuf";
import { ListModelSourcesResponseSchema, SourceSignIn } from "../src/gen/turbo_training_pb.js";
import { readOnly, redirected, signInText } from "../src/sourcetext.js";

test("each sign-in reads in plain words", () => {
  assert.equal(signInText(SourceSignIn.GOOGLE), "Google sign-in");
  assert.equal(signInText(SourceSignIn.API_KEY), "API key");
  assert.equal(signInText(SourceSignIn.ENVIRONMENT), "the server's own");
  assert.equal(signInText(SourceSignIn.UNSPECIFIED), "");
});

test("the page changes sources only for a user on an admin or zoo server", () => {
  assert.equal(readOnly(create(ListModelSourcesResponseSchema, {})), "read-only: the demo was started without --user");
  assert.equal(
    readOnly(create(ListModelSourcesResponseSchema, { user: "alice" })),
    "read-only: the server was started without --admin or --zoo",
  );
  assert.equal(readOnly(create(ListModelSourcesResponseSchema, { user: "alice", admin: true })), "");
});

test("a redirect brings a code and a state, or an error", () => {
  assert.deepEqual(redirected("?state=s%2B1&code=4%2F0Ab"), { code: "4/0Ab", state: "s+1", error: "" });
  assert.deepEqual(redirected("?state=s1&error=access_denied"), { code: "", state: "s1", error: "access_denied" });
  assert.equal(redirected("?code=only"), null);
  assert.equal(redirected(""), null);
});
