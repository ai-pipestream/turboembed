# Demo page layout receipt (#2076)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` at three tiers and
`testdata/tiny-modernbert-bundle`, and `turbo-kserve-demo` in front of
it (`scripts/e2e-web.sh`, docs/demo.md). Machine: a 4-core x86_64 Linux
container, no GPU.

- `red.log`: `demo/web/e2e/page.test.ts` against the page as it was on
  main at ca6264e0. All three tests fail; the two layout tests list 19
  faults at 1280 px and 22 at 390 px.
- `green.log`: the same run with the change. All three pass.

Each picture is the same tab before (left) and after (right), full page
from the top, taken by the same run:

| File | Width | What to look at |
|---|---|---|
| `desktop-vocabulary.png` | 1280 px | The format select and Import button inside the box; checkboxes beside their labels; "None." for the empty lists. |
| `desktop-analyze.png` | 1280 px | Option fields that show their values; the export format select inside the box; the header on one row. |
| `desktop-semantic.png` | 1280 px | No empty cards under the neighbours. |
| `phone-vocabulary.png` | 390 px | The page no longer 762 px wide. |
| `phone-analyze.png` | 390 px | The page no longer 588 px wide. |
