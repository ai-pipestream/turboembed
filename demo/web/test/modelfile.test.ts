// The Models tab's non-commercial line and its check of a saved file.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mismatch, nonCommercial } from "../src/modelfile.js";

test("non-commercial by the artifact's own licence or by its training data", () => {
  assert.equal(nonCommercial({ licence: "CC-BY-NC-4.0", trainedOn: [] }), true);
  assert.equal(nonCommercial({ licence: "CC-BY-NC-SA-4.0", trainedOn: [] }), true);
  assert.equal(nonCommercial({ licence: "Apache-2.0", trainedOn: [] }), false);
  assert.equal(nonCommercial({ licence: "CC-BY-SA-4.0", trainedOn: [] }), false);
  const data = (licence: string) => ({ name: "corpus", licence });
  assert.equal(nonCommercial({ licence: "Apache-2.0", trainedOn: [data("CC-BY-4.0"), data("CC-BY-NC-SA-3.0")] }), true);
  assert.equal(nonCommercial({ licence: "MIT", trainedOn: [data("CC-BY-4.0")] }), false);
  assert.equal(nonCommercial({ licence: "LicenseRef-CC-BY-NCX", trainedOn: [] }), false);
});

test("a saved file must be the size and SHA-256 the catalog lists", () => {
  const f = { path: "model/weights.bin", size: 3n, sha256: "ab".repeat(32) };
  assert.equal(mismatch(f, 3, "ab".repeat(32)), "");
  assert.equal(mismatch(f, 2, "ab".repeat(32)), "model/weights.bin: 2 bytes arrived, the catalog lists 3");
  assert.equal(
    mismatch(f, 3, "cd".repeat(32)),
    `model/weights.bin: the bytes have SHA-256 ${"cd".repeat(32)}, the catalog lists ${"ab".repeat(32)}`,
  );
});
