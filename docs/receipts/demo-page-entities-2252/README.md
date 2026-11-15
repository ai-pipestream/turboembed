# Entity views receipt (#2252)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle`, and `turbo-kserve-demo` in front
of it. The document is three paragraphs about railway lines between
ten European cities, analyzed with `default-v1`, whose place-name step
writes the entity layer (24 mentions). Machine: a 4-core x86_64 Linux
container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. The same checks are in the
repository as `demo/web/e2e/entities.test.ts`, and the unit tests for
the arithmetic are in `demo/web/test/entities.test.ts`; both run in CI.

- `red.log`: the checks against the page before this change, which has
  no entity views. 0 of 13 pass.
- `green.log`: the same checks with the change. All 13 pass.

| File | What to look at |
|---|---|
| `entities.png` | The panel with Paris pointed at: its node, the entities it shares a sentence with, and its row in the strips. |
| `mention-strips.png` | A mention of Berlin pointed at: its sentence under the strips. |

Not shown: relation arrows. No test bundle in the repository carries a
relation model, so no run here produced a relation layer. Turning
relation annotations into edges between entities is covered by the unit
tests (`link`), and the drawing of an arrow with its label was not
exercised against a server.
