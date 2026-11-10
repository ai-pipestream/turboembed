// The catalog filter: kinds in plain words, words matched in any case
// across id, kind, language and licence.

import assert from "node:assert/strict";
import { test } from "node:test";
import { kindName, kinds, matches, plain, words } from "../src/catalogfilter.js";

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
