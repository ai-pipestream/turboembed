// The catalog filter: kinds in plain words, words matched in any case
// across id, kind, language and licence.

import assert from "node:assert/strict";
import { test } from "node:test";
import { kindName, kinds, matches, plain, purpose, words } from "../src/catalogfilter.js";

const entries = [
  { id: "baai-bge-small-en-v1.5", kind: "KIND_MODEL", language: "en", licence: "MIT" },
  { id: "ud-spanish-ancora-r2.16", kind: "KIND_DATASET", language: "es", licence: "CC-BY-4.0" },
  { id: "turbo-en-goemotions-head-1", kind: "KIND_HEAD", language: "en", licence: "Apache-2.0" },
  { id: "symspell-frequency-dictionary", kind: "KIND_LEXICON", language: "en", licence: "MIT" },
  { id: "Straße-École", kind: "KIND_MODEL", language: "mul", licence: "MIT" },
];

test("kinds and enum values read as plain words", () => {
  assert.equal(kindName("KIND_MODEL"), "model");
  assert.equal(kindName("KIND_STATIC_MODEL"), "static model");
  assert.equal(kindName("other"), "other");
  assert.equal(plain("TASK_EXTRACTIVE_QA", "TASK_"), "extractive qa");
  assert.equal(plain("PROMPT_QUERY", "PROMPT_"), "query");
});

test("typed words are split on spaces and lower cased", () => {
  assert.deepEqual(words("  BGE   small "), ["bge", "small"]);
  assert.deepEqual(words(""), []);
});

test("any white space parts two words", () => {
  assert.deepEqual(words("bge\tsmall"), ["bge", "small"]);
  assert.deepEqual(words("bge\u00a0small"), ["bge", "small"]);
  assert.deepEqual(words("bge\u3000small\n"), ["bge", "small"]);
  assert.deepEqual(words("\u2003\t "), []);
  assert.deepEqual(words("bge\u0085small"), ["bge", "small"]);
  const white = "\t\n\u000b\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
  for (const ch of white) assert.deepEqual(words(`bge${ch}small`), ["bge", "small"], `U+${ch.codePointAt(0)!.toString(16)}`);
  const hit = (text: string) => entries.filter((e) => matches(e, "", text)).map((e) => e.id);
  assert.deepEqual(hit("bge\tsmall"), ["baai-bge-small-en-v1.5"]);
  assert.deepEqual(hit("bge\u00a0small"), ["baai-bge-small-en-v1.5"]);
});

test("the fold matches sharp s, a dotted capital I and astral letters", () => {
  assert.deepEqual(words("Straße"), words("STRASSE"));
  assert.deepEqual(words("\u0130stanbul"), ["istanbul"]);
  assert.deepEqual(words("\u{10400}"), ["\u{10428}"]);
  const hit = (text: string) => [...entries, { id: "\u0130stanbul-ner", kind: "KIND_HEAD", language: "tr", licence: "MIT" }]
    .filter((e) => matches(e, "", text)).map((e) => e.id);
  assert.deepEqual(hit("STRASSE"), ["Straße-École"]);
  assert.deepEqual(hit("straße"), ["Straße-École"]);
  assert.deepEqual(hit("istanbul"), ["\u0130stanbul-ner"]);
});

test("the fold treats both sharp s letters and both sigmas alike, and is stable", () => {
  const one = (id: string, text: string) => matches({ id, kind: "KIND_MODEL", language: "de", licence: "MIT" }, "", text);
  for (const id of ["Straße", "STRA\u1e9eE", "STRASSE"]) {
    for (const text of ["Straße", "STRA\u1e9eE", "strasse", "STRASSE"]) assert.equal(one(id, text), true, `${id} by ${text}`);
  }
  assert.equal(one("\u039f\u0394\u039f\u03a3", "\u03bf\u03b4\u03bf\u03c3"), true);
  assert.equal(one("\u03bf\u03b4\u03bf\u03c2-ner", "\u039f\u0394\u039f\u03a3"), true);
  for (const text of ["\u1e9e", "ß", "\u0130", "I", "\u03a3", "\u03c2", "\u{10400}", "Straße \u0130stanbul"]) {
    const once = words(text);
    assert.deepEqual(words(once.join(" ")), once, text);
  }
});

test("a kind matches in its enum spelling too", () => {
  const hit = (text: string) => entries.filter((e) => matches(e, "", text)).map((e) => e.id);
  assert.deepEqual(hit("KIND_LEXICON"), ["symspell-frequency-dictionary"]);
  assert.deepEqual(hit("kind_dataset es"), ["ud-spanish-ancora-r2.16"]);
});

test("every word must match, in id, kind, language or licence", () => {
  const ids = (kind: string, text: string) => entries.filter((e) => matches(e, kind, text)).map((e) => e.id);
  assert.equal(ids("", "").length, entries.length);
  assert.deepEqual(ids("", "BGE small"), ["baai-bge-small-en-v1.5"]);
  assert.deepEqual(ids("", "dataset"), ["ud-spanish-ancora-r2.16"]);
  assert.deepEqual(ids("", "mit lexicon"), ["symspell-frequency-dictionary"]);
  assert.deepEqual(ids("KIND_MODEL", "mit"), ["baai-bge-small-en-v1.5", "Straße-École"]);
  assert.deepEqual(ids("KIND_HEAD", "mit"), []);
  assert.deepEqual(ids("", "école"), ["Straße-École"]);
  assert.deepEqual(ids("", "cc-by es"), ["ud-spanish-ancora-r2.16"]);
});

test("kinds are counted in the order first seen", () => {
  assert.deepEqual(kinds(entries), [["KIND_MODEL", 2], ["KIND_DATASET", 1], ["KIND_HEAD", 1], ["KIND_LEXICON", 1]]);
});

test("an entry's purpose comes from its component, else its kind, and the filter reads it", () => {
  const e = (kind: string, component = "") => ({ id: "x", kind, language: "en", licence: "MIT", component });
  assert.equal(purpose(e("KIND_MODEL", "COMPONENT_DENSE_ENCODER")), "Embeds text for semantic search");
  assert.equal(purpose(e("KIND_HEAD", "COMPONENT_POS")), "Tags parts of speech");
  assert.equal(purpose(e("KIND_MODEL", "COMPONENT_SOMETHING_NEW")), "Something new");
  assert.equal(purpose(e("KIND_DATASET")), "Data to train and evaluate with");
  assert.equal(purpose(e("KIND_OTHER_THING")), "Other thing");
  assert.equal(purpose({ id: "x", kind: "KIND_LEXICON", language: "en", licence: "MIT" }), "Word list");
  assert.equal(matches(e("KIND_MODEL", "COMPONENT_DENSE_ENCODER"), "", "semantic search"), true);
  assert.equal(matches(e("KIND_MODEL", "COMPONENT_EXTRACTIVE_QA"), "", "semantic search"), false);
});
