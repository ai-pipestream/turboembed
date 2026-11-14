import { test } from "node:test";
import assert from "node:assert/strict";
import { csv, field, fileName, inert, tsv } from "../src/csv.js";

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

test("a value a sheet would read as a formula is made text; a number is not", () => {
  assert.equal(inert("=SUM(A1:A9)"), "'=SUM(A1:A9)");
  assert.equal(inert("+1 and more"), "'+1 and more");
  assert.equal(inert("-cmd"), "'-cmd");
  assert.equal(inert("@here"), "'@here");
  assert.equal(inert("\t=1+1"), "'\t=1+1");
  assert.equal(inert("-0.25"), "-0.25");
  assert.equal(inert("+3"), "+3");
  assert.equal(inert("-1e-3"), "-1e-3");
  assert.equal(inert("plain = text"), "plain = text");
  assert.equal(field('=HYPERLINK("x","y")'), `"'=HYPERLINK(""x"",""y"")"`);
  assert.equal(tsv([["=1+1", "-0.5"]]), "'=1+1\t-0.5");
});
