# Similarity matrix and query spotlight receipt (#2250, #2253)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle`, and `turbo-kserve-demo` in front
of it. The document is four headed sections (computing history, money,
vectors, railways), analyzed with `default-v1`: ten chunks under four
coarser units. Machine: a 4-core x86_64 Linux container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. Browser tests for
`demo/web/e2e` are follow-up work. The unit tests for the arithmetic
(`demo/web/test/vizmatrix.test.ts`) are in the repository and run in CI.

- `red.log`: the checks against the page before this change, which has
  neither view. 1 of 20 passes (no script error).
- `green.log`: the same checks with the change. All 20 pass.

| File | What to look at |
|---|---|
| `matrix.png` | The matrix with the cell for chunks 3 and 7 pointed at, and the four blocks at the chunk cuts of the coarser units. |
| `spotlight.png` | A query placed on the map: the diamond, the glow, the lines to the closest, the list, and the marked chunks in the text. |
| `dark.png` | The same in the dark theme, where the ramp runs from dark to light. |
| `phone.png` | The matrix at 390 pixels wide. |

What the pictures do not show: that the closest chunks are the right
ones. The test bundle is a tiny model with random weights, so its
cosines sit between 0.95 and 0.99 and say nothing about meaning; the
checks hold the page to the server's numbers (order, count, marks), and
the server's numbers are held to the stored vectors in
`server/tests/inspect_pairwise_query.rs`.

Rows are chunks, not sentences: the served profiles write chunk vectors
only. A document over 512 chunks was not run here.
