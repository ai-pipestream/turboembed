// A typed annotation reads as its set fields, by name, through the
// generated descriptors.

import { test } from "node:test";
import assert from "node:assert/strict";
import { create } from "@bufbuild/protobuf";
import { LayerWireStatus, TypedAnnotationSchema } from "../src/gen/turbo_typed_annotations_pb.js";
import { LayerInfoSchema } from "../src/gen/turbo_analysis_pb.js";
import { brief } from "../src/brief.js";

test("an entity with its votes", () => {
  const a = create(TypedAnnotationSchema, {
    spans: [{ start: 0, end: 12 }],
    probability: 0.875,
    value: {
      case: "entity",
      value: {
        kind: "PERSON",
        tokenStart: 0,
        tokenEnd: 2,
        canonical: "Ada Lovelace",
        votes: [{ annotator: "opennlp", probability: 0.9 }, { annotator: "rules" }],
      },
    },
  });
  assert.equal(
    brief(TypedAnnotationSchema, a, 0, ["spans", "probability"]),
    'entity {kind "PERSON", tokenEnd 2, canonical "Ada Lovelace", votes [2] {annotator "opennlp", probability 0.9} {annotator "rules"}}',
  );
});

test("enums by name, unset fields left out", () => {
  const l = create(LayerInfoSchema, { layer: "turbo:tokens", status: LayerWireStatus.SKIPPED, items: 3 });
  assert.equal(brief(LayerInfoSchema, l), 'layer "turbo:tokens", status SKIPPED, items 3');
});
