import { test } from "node:test";
import assert from "node:assert/strict";
import { recall, recallList, remember, storable, stored, TABS } from "../src/keep.js";

test("text, numbers, checkboxes and selects are kept; a password and a file never are", () => {
  assert.equal(storable("TEXTAREA", "textarea"), true);
  assert.equal(storable("SELECT", "select-one"), true);
  assert.equal(storable("INPUT", "text"), true);
  assert.equal(storable("INPUT", "number"), true);
  assert.equal(storable("INPUT", "checkbox"), true);
  assert.equal(storable("INPUT", "password"), false);
  assert.equal(storable("INPUT", "PASSWORD"), false);
  assert.equal(storable("INPUT", "file"), false);
  assert.equal(storable("BUTTON", "submit"), false);
});

test("a checkbox is stored by its state, anything else by its text", () => {
  assert.equal(stored("checkbox", "on", true), "1");
  assert.equal(stored("checkbox", "on", false), "0");
  assert.equal(stored("text", "Grüße, 世界", false), "Grüße, 世界");
});

test("the tabs that hold keys and addresses keep nothing", () => {
  assert.equal(TABS.includes("models"), false);
  assert.equal(TABS.includes("server"), false);
});

test("with no storage in reach, nothing is kept and nothing throws", () => {
  remember("x", "y");
  assert.equal(recall("x"), null);
  assert.equal(recallList("x"), null);
});
