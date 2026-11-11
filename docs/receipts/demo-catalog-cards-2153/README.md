# Demo Models catalog cards receipt (#2153)

The Models tab in Chromium through Playwright, against `turbo-kserve` on
the CPU with `testdata/tiny-bert-bundle` at three tiers and
`testdata/tiny-modernbert-bundle`, and `turbo-kserve-demo` in front of
it (`scripts/e2e-web.sh`, docs/demo.md). The catalog is the built-in
one: 46 entries. Machine: a 4-core x86_64 Linux container, no GPU.

- `red.log`: `demo/web/e2e/page.test.ts` with the card checks, against
  the page as it was on main. The catalog test fails: the entries are
  one to a row and none says what it is for. The other three pass.
- `green.log`: the same run with the change. All four pass.

The pictures are from a demo started with `--user`, after "Read the
catalog". The paired ones show before (left) and after (right):

| File | Width | What to look at |
|---|---|---|
| `desktop-models.png` | 1280 px | Three cards to a row; under each name a line saying what the entry is for; Install filled in where no terms are left to accept. |
| `desktop-models-filtered.png` | 1280 px | After only: "bge" typed, 6 of 46 entries left as cards. |
| `phone-models.png` | 390 px | One card to a row, inside the catalog's box. |
