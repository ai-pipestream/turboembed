# Document map receipt (#2246, #2247, #2251)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle`, and `turbo-kserve-demo` in front
of it. The document is four paragraphs (computing history, then banks
and interest rates, then vectors), analyzed with `default-v1`, which
cut it into five chunks. Machine: a 4-core x86_64 Linux container, no
GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. Browser tests for
`demo/web/e2e` are follow-up work. The unit tests for the arithmetic
(`demo/web/test/vizmodel.test.ts`) are in the repository and run in CI.

- `red.log`: the checks against the page before this change (a canvas
  scatter, a drift canvas and stack bars, in a closed panel). 0 of 24
  pass.
- `green.log`: the same checks with the change. All 24 pass.

| File | What to look at |
|---|---|
| `before.png` | The panel before: closed until opened, then two canvases and bars. |
| `document-map.png` | The whole panel after: the meaning map with the text, the seismograph, the river. |
| `document-map-dark.png` | The same in the dark scheme, with its own steps of each colour. |
| `meaning-map.png` | Chunk 2 pointed at: its dot, its terms, and its text lit. |
| `seismograph.png` | A gap pointed at: the sentence on each side. The test model writes no sentence vectors, so the stems are the chunker's cost; the note above the figure says so. |
| `term-river.png` | "cobol" pointed at: the other streams fade. |
| `term-marked-in-text.png` | Its two occurrences marked in the text. |
| `phone.png` | At 390 px: each figure scrolls inside its own box; the page does not scroll sideways. |

The tiny test model's vectors carry little meaning, so the map's shape
here shows the drawing, not the model.
