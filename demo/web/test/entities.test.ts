import { test } from "node:test";
import assert from "node:assert/strict";
import { finest, gather, kindSlots, layout, link, sentenceAt, together, type Mention } from "../src/viz/entities.js";

const m = (text: string, kind: string, start: number, canonical = ""): Mention => ({ start, end: start + text.length, kind, canonical, text });

test("mentions of one name and kind are one entity, whatever the case or composition", () => {
  const all = [m("Köln", "location", 0), m("London", "location", 10), m("KÖLN", "location", 20), m("London", "person", 30), m("köln", "location", 40)];
  const { entities, of } = gather(all);
  assert.deepEqual(entities.map((e) => [e.name, e.kind, e.mentions.length]), [["Köln", "location", 3], ["London", "location", 1], ["London", "person", 1]]);
  assert.deepEqual(of, [0, 1, 0, 2, 0]);
});

test("a canonical name gathers different spellings and names the entity", () => {
  const { entities } = gather([m("Lovelace", "person", 0, "Ada Lovelace"), m("Ada Lovelace", "person", 20, "Ada Lovelace")]);
  assert.deepEqual(entities.map((e) => [e.name, e.mentions]), [["Ada Lovelace", [0, 1]]]);
});

test("relations between mentions become edges between entities, counted once per pair and kind", () => {
  const of = [0, 1, 0, 2];
  const edges = link([{ subject: 0, object: 1, kind: "works for", score: 0.6 }, { subject: 2, object: 1, kind: "works for", score: 0.9 }, { subject: 0, object: 2, kind: "is" }, { subject: 3, object: 9, kind: "lost" }], of);
  assert.deepEqual(edges, [{ from: 0, to: 1, kind: "works for", support: 2, score: 0.9 }]);
});

test("the layout is the same each time, inside its box, with linked nodes nearer than unlinked ones", () => {
  const edges = [{ from: 0, to: 1 }];
  const a = layout(4, edges, 600, 300);
  assert.deepEqual(a, layout(4, edges, 600, 300));
  assert.equal(a.every(([x, y]) => x >= 40 && x <= 560 && y >= 40 && y <= 260), true);
  const d = (i: number, j: number): number => Math.hypot(a[i][0] - a[j][0], a[i][1] - a[j][1]);
  assert.equal(d(0, 1) < d(2, 3), true);
  assert.deepEqual(layout(1, [], 600, 300), [[300, 150]]);
  assert.deepEqual(layout(0, [], 600, 300), []);
});

test("the finest spans are those that hold no other", () => {
  const got = finest([{ start: 0, end: 100 }, { start: 0, end: 40 }, { start: 40, end: 100 }, { start: 40, end: 100 }, { start: 5, end: 5 }]);
  assert.deepEqual(got, [{ start: 0, end: 40 }, { start: 40, end: 100 }]);
  assert.deepEqual(finest([{ start: 0, end: 9 }]), [{ start: 0, end: 9 }]);
});

test("a mention shows its sentence, or the text around it when no sentence holds it", () => {
  const text = "One here. Two there.";
  assert.equal(sentenceAt(text, [{ start: 0, end: 9 }, { start: 10, end: 20 }], 14), "Two there.");
  assert.equal(sentenceAt(text, [], 14), text);
});

test("kinds keep the slot of their first appearance, and past the cap share the last", () => {
  const s = kindSlots(["person", "location", "person", "work", "org"], 3);
  assert.deepEqual([...s], [["person", 0], ["location", 1], ["work", 2], ["org", 2]]);
});

test("entities in one sentence go together, counted by sentence", () => {
  const all = [m("Paris", "location", 0), m("Berlin", "location", 10), m("Paris", "location", 30), m("Berlin", "location", 40), m("Rome", "location", 45), m("Rome", "location", 70)];
  const { of } = gather(all);
  const pairs = together(all, of, [{ start: 0, end: 20 }, { start: 25, end: 60 }, { start: 60, end: 80 }]);
  const named = pairs.map((p) => `${p.from}-${p.to}:${p.count}`).sort();
  assert.deepEqual(named, ["0-1:2", "0-2:1", "1-2:1"]);
  assert.deepEqual(together(all, of, []), []);
});

test("an entity named twice in one sentence is counted once for that sentence", () => {
  // Paris twice and Berlin once in the first sentence, Paris twice alone in the second.
  const all = [m("Paris", "location", 0), m("Paris", "location", 8), m("Berlin", "location", 16), m("Paris", "location", 30), m("Paris", "location", 40)];
  const { of } = gather(all);
  const pairs = together(all, of, [{ start: 0, end: 25 }, { start: 25, end: 50 }]);
  assert.deepEqual(pairs, [{ from: 0, to: 1, count: 1 }]);
});
