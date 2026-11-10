# Demo Models catalog receipt (#2079)

The Models tab in Chromium through Playwright, against `turbo-kserve` on
the CPU with `testdata/tiny-bert-bundle` at three tiers and
`testdata/tiny-modernbert-bundle`, and `turbo-kserve-demo` in front of
it (`scripts/e2e-web.sh`, docs/demo.md). The catalog is the built-in
one: 46 entries. Machine: a 4-core x86_64 Linux container, no GPU.

- `red.log`: `demo/web/e2e/page.test.ts` with the new catalog test,
  against the page as it was on main at 0624d557. The catalog test and
  the two layout tests fail waiting for the filter field, which that
  page does not have; the embed test passes.
- `green.log`: the same run with the change. All four pass.

The pictures are from a demo started with `--user`, after "Read the
catalog". The paired ones show before (left) and after (right), from
the top of the page:

| File | Width | What to look at |
|---|---|---|
| `desktop-models.png` | 1280 px | One line of facts per entry with the id unbroken; `model` in place of `KIND_MODEL`; files behind "2 files"; the filter, the count and a chip per kind; Installed and Accepted terms above the catalog. |
| `desktop-models-filtered.png` | 1280 px | After only: "bge" typed, 6 of 46 entries left. |
| `phone-models.png` | 390 px | The entries wrap in the card; before, the table ran past its right edge. |
| `desktop-server.png` | 1280 px | Task and roles as plain words in place of `TASK_EMBED` and `PROMPT_QUERY`. |
