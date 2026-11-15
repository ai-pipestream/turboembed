// The annotated text's colours: by label for layers that name things,
// by layer and instance for layers that cut the text.

import assert from "node:assert/strict";
import { test } from "node:test";
import { colourOf, found, kept, layerKey, layerName, leadOf, legend, painted, rank, type Ann, type Seg } from "../src/marks.js";

const ann = (layer: string, item: number, label = "", instance = ""): Ann => ({ layer, instance, item, label, score: 1 });
const none = new Set<string>();

const sentence = ann("turbo:sentences", 0, "S0");
const token = (i: number) => ann("turbo:tokens", i, "token");
const person = ann("turbo:entities", 0, "person");
const place = ann("turbo:entities", 1, "location");

const doc: Seg[] = [
  { text: "Ada", annotations: [sentence, token(0), person] },
  { text: " ", annotations: [sentence, person] },
  { text: "Lovelace", annotations: [sentence, token(1), person] },
  { text: " lived in ", annotations: [sentence] },
  { text: "London", annotations: [sentence, token(4), place] },
];

test("two instances of one layer have different keys and names", () => {
  assert.equal(layerKey("turbo:boundaries", "") === layerKey("turbo:boundaries", "lexical"), false);
  assert.equal(layerName("turbo:boundaries", ""), "turbo:boundaries");
  assert.equal(layerName("turbo:boundaries", "lexical"), "turbo:boundaries [lexical]");
});

test("the filter keeps one instance without the other", () => {
  const a = ann("turbo:boundaries", 0);
  const b = ann("turbo:boundaries", 0, "", "lexical");
  assert.deepEqual(kept([a, b], new Set([layerKey("turbo:boundaries", "lexical")])), [b]);
  assert.deepEqual(kept([a, b], none), [a, b]);
});

test("a naming layer is coloured by label, a cutting layer by layer", () => {
  assert.deepEqual(colourOf(person), { key: "label:person", name: "person" });
  assert.deepEqual(colourOf(sentence), { key: "layer:turbo:sentences#", name: "turbo:sentences" });
  assert.equal(rank(person) < rank(token(0)) && rank(token(0)) < rank(sentence), true);
});

test("with no filter only the named things paint", () => {
  assert.equal(leadOf(doc), "naming");
  assert.deepEqual(painted(doc[0].annotations, none, "naming"), [person]);
  assert.deepEqual(painted(doc[3].annotations, none, "naming"), []);
});

test("a document that names nothing paints its sentences", () => {
  const plain: Seg[] = [{ text: "Ada", annotations: [sentence, token(0)] }];
  assert.equal(leadOf(plain), "sentences");
  assert.deepEqual(painted(plain[0].annotations, none, "sentences"), [sentence]);
});

test("a document with neither names nor sentences paints all it has", () => {
  const bare: Seg[] = [{ text: "Ada", annotations: [token(0)] }];
  assert.equal(leadOf(bare), "all");
  assert.deepEqual(painted(bare[0].annotations, none, "all"), [token(0)]);
});

test("picked layers paint with the finer one leading", () => {
  const picked = new Set([layerKey("turbo:sentences", ""), layerKey("turbo:tokens", "")]);
  assert.deepEqual(painted(doc[0].annotations, picked, "naming"), [token(0), sentence]);
});

test("the legend counts items, not stretches, in order of first appearance", () => {
  assert.deepEqual(legend(doc, none), [
    { key: "label:person", name: "person", index: 0, items: 1 },
    { key: "label:location", name: "location", index: 1, items: 1 },
  ]);
  const tokens = legend(doc, new Set([layerKey("turbo:tokens", "")]));
  assert.deepEqual(tokens, [{ key: "layer:turbo:tokens#", name: "turbo:tokens", index: 0, items: 3 }]);
});

test("what was found is listed by label with each item's whole text", () => {
  assert.deepEqual(found(doc), [
    { label: "person", texts: ["Ada Lovelace"] },
    { label: "location", texts: ["London"] },
  ]);
  assert.deepEqual(found([{ text: "Ada", annotations: [sentence] }]), []);
});
