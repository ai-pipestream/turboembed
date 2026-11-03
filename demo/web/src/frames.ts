// Upload bodies for the demo's PUT endpoints (docs/demo.md, Uploads):
// Connect envelope frames of binary protobuf built with the generated
// types, as Blobs, so a file is streamed from disk and never read into
// memory. A file's bytes become a string field by writing the field's
// tag and length in front of them.

import { create, toBinary, type MessageInitShape } from "@bufbuild/protobuf";
import {
  ImportDictionaryHeaderSchema,
  LearnVocabularyDocumentSchema,
  LearnVocabularyHeaderSchema,
} from "./gen/turbo_vocabulary_pb.js";
import { IndexDocumentRequestSchema } from "./gen/turbo_search_pb.js";

/** The demo's per-frame limit: the server's 64 MiB message limit less room for wrapping. */
export const MAX_FRAME_BYTES = (64 << 20) - 16;

/** The demo's 2 GiB cap on one upload. */
export const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;

/** Bytes the Blob constructor takes. */
type Bytes = Uint8Array<ArrayBuffer>;

/** A message's binary form; protobuf-es always writes to a fresh ArrayBuffer. */
function bytes(b: Uint8Array): Bytes {
  return b as Bytes;
}

/** A document's text, typed in or a file. */
export type Text = string | Blob;

/** The five-byte envelope prefix of an uncompressed frame of `n` bytes. */
export function prefix(n: number): Bytes {
  if (n > MAX_FRAME_BYTES) throw new RangeError(`a frame of ${n} bytes is over the ${MAX_FRAME_BYTES} byte limit`);
  const p = new Uint8Array(5);
  new DataView(p.buffer).setUint32(1, n, false);
  return p;
}

/** One frame holding `message`. */
export function envelope(message: Uint8Array): Bytes {
  const out = new Uint8Array(5 + message.length);
  out.set(prefix(message.length));
  out.set(message, 5);
  return out;
}

/** `n` as a protobuf varint. */
export function varint(n: number): Bytes {
  const out: number[] = [];
  let v = n;
  while (v >= 0x80) {
    out.push((v % 0x80) | 0x80);
    v = Math.floor(v / 0x80);
  }
  out.push(v);
  return Uint8Array.from(out);
}

/** The tag and length of length-delimited field `field` holding `n` bytes. */
function fieldHead(field: number, n: number): Bytes {
  const tag = varint(field * 8 + 2);
  const len = varint(n);
  const out = new Uint8Array(tag.length + len.length);
  out.set(tag);
  out.set(len, tag.length);
  return out;
}

/** One frame: `head` (a message's bytes, possibly empty), then field `field` holding `text`. */
function textFrame(head: Bytes, field: number, text: Blob): BlobPart[] {
  const fh = fieldHead(field, text.size);
  return [prefix(head.length + fh.length + text.size), head, fh, text];
}

/** Each part's size in all, refused over the 2 GiB cap. */
function body(parts: BlobPart[]): Blob {
  const b = new Blob(parts, { type: "application/octet-stream" });
  if (b.size > MAX_UPLOAD_BYTES) throw new RangeError(`the upload is ${b.size} bytes, over the 2 GiB cap`);
  return b;
}

/** ImportDictionary: the header frame, then the dictionary's raw bytes. */
export function importBody(header: MessageInitShape<typeof ImportDictionaryHeaderSchema>, file: Blob): Blob {
  return body([envelope(toBinary(ImportDictionaryHeaderSchema, create(ImportDictionaryHeaderSchema, header))), file]);
}

/** LearnVocabulary: the header frame, then one document frame per text. */
export function learnBody(header: MessageInitShape<typeof LearnVocabularyHeaderSchema>, docs: Text[]): Blob {
  const parts: BlobPart[] = [envelope(toBinary(LearnVocabularyHeaderSchema, create(LearnVocabularyHeaderSchema, header)))];
  for (const d of docs) {
    if (typeof d === "string") {
      parts.push(envelope(toBinary(LearnVocabularyDocumentSchema, create(LearnVocabularyDocumentSchema, { text: d }))));
    } else {
      parts.push(...textFrame(new Uint8Array(0), 1, d));
    }
  }
  return body(parts);
}

/** IndexDocuments: one IndexDocumentRequest frame per document. */
export function indexBody(index: string, docs: { id: string; text: Text }[]): Blob {
  const parts: BlobPart[] = [];
  for (const d of docs) {
    if (typeof d.text === "string") {
      const m = create(IndexDocumentRequestSchema, { indexName: index, docId: d.id, text: d.text });
      parts.push(envelope(toBinary(IndexDocumentRequestSchema, m)));
    } else {
      const head = bytes(toBinary(IndexDocumentRequestSchema, create(IndexDocumentRequestSchema, { indexName: index, docId: d.id })));
      parts.push(...textFrame(head, 3, d.text));
    }
  }
  return body(parts);
}
