// The upload bodies decode, frame by frame, to the messages they stand
// for, whether a text was typed or is a file.

import { test } from "node:test";
import assert from "node:assert/strict";
import { fromBinary } from "@bufbuild/protobuf";
import { ImportDictionaryHeaderSchema, LearnVocabularyDocumentSchema, LearnVocabularyHeaderSchema } from "../src/gen/turbo_vocabulary_pb.js";
import { IndexDocumentRequestSchema } from "../src/gen/turbo_search_pb.js";
import { MAX_FRAME_BYTES, envelope, importBody, indexBody, learnBody, prefix, varint } from "../src/frames.js";

async function frames(b: Blob): Promise<Uint8Array[]> {
  const all = new Uint8Array(await b.arrayBuffer());
  const out: Uint8Array[] = [];
  let at = 0;
  while (at < all.length) {
    assert.equal(all[at], 0, "an uncompressed frame");
    const n = new DataView(all.buffer, at + 1, 4).getUint32(0, false);
    out.push(all.subarray(at + 5, at + 5 + n));
    at += 5 + n;
  }
  assert.equal(at, all.length, "no bytes after the last frame");
  return out;
}

const file = (s: string) => new Blob([s]);

test("varints", () => {
  assert.deepEqual([...varint(0)], [0]);
  assert.deepEqual([...varint(127)], [127]);
  assert.deepEqual([...varint(300)], [0xac, 0x02]);
  assert.deepEqual([...varint(2 ** 31)], [0x80, 0x80, 0x80, 0x80, 0x08]);
});

test("a frame over the limit is refused", () => {
  assert.throws(() => prefix(MAX_FRAME_BYTES + 1), RangeError);
  assert.deepEqual([...envelope(new Uint8Array([9]))], [0, 0, 0, 0, 1, 9]);
});

test("learn: the header, then typed and file documents", async () => {
  const long = "word ".repeat(40);
  const f = await frames(learnBody({ minTermFrequency: 2, lowercase: true }, ["typed text", file("from a file \u00e9"), file(long)]));
  assert.equal(f.length, 4);
  const h = fromBinary(LearnVocabularyHeaderSchema, f[0]);
  assert.equal(h.minTermFrequency, 2);
  assert.equal(h.lowercase, true);
  assert.deepEqual(f.slice(1).map((x) => fromBinary(LearnVocabularyDocumentSchema, x).text), ["typed text", "from a file \u00e9", long]);
});

test("index: one request per document, the file as its text", async () => {
  const f = await frames(indexBody("idx", [{ id: "a", text: "typed" }, { id: "b.txt", text: file("x".repeat(200)) }]));
  const docs = f.map((x) => fromBinary(IndexDocumentRequestSchema, x));
  assert.deepEqual(docs.map((d) => [d.indexName, d.docId, d.text]), [["idx", "a", "typed"], ["idx", "b.txt", "x".repeat(200)]]);
});

test("import: the header frame, then the raw file", async () => {
  const raw = "word\tcount\nhello\t3\n";
  const all = new Uint8Array(await importBody({ formatId: "tsv", name: "d", byteSize: 19n, sha256: "ab" }, file(raw)).arrayBuffer());
  const n = new DataView(all.buffer, 1, 4).getUint32(0, false);
  const h = fromBinary(ImportDictionaryHeaderSchema, all.subarray(5, 5 + n));
  assert.equal(h.formatId, "tsv");
  assert.equal(h.byteSize, 19n);
  assert.equal(new TextDecoder().decode(all.subarray(5 + n)), raw);
});
