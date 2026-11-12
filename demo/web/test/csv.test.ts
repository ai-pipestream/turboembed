import { test } from "node:test";
import assert from "node:assert/strict";
import { csv, field, fileName, tsv } from "../src/csv.js";

test("a plain field is left alone; a comma, a quote or a line break quotes it", () => {
  assert.equal(field("0.9731"), "0.9731");
  assert.equal(field("Grüße 世界"), "Grüße 世界");
  assert.equal(field("a, b"), '"a, b"');
  assert.equal(field('say "hi"'), '"say ""hi"""');
  assert.equal(field("two\nlines"), '"two\nlines"');
  assert.equal(field(" padded"), '" padded"');
});

test("rows end CR LF, the last one too", () => {
  assert.equal(csv([["a", "b"], ["1", "x,y"]]), 'a,b\r\n1,"x,y"\r\n');
  assert.equal(csv([]), "");
});

test("the clipboard form keeps one value per cell", () => {
  assert.equal(tsv([["a", "b\tc"], ["two\r\nlines", "d"]]), "a\tb c\ntwo lines\td");
});

test("a file name keeps letters of any script and digits", () => {
  assert.equal(fileName("Passage similarity (tiny-bert-bundle)"), "Passage-similarity-tiny-bert-bundle.csv");
  assert.equal(fileName("Ähnlichkeit 世界 2"), "Ähnlichkeit-世界-2.csv");
  assert.equal(fileName("///"), "result.csv");
});
