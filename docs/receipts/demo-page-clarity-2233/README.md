# Demo page clarity receipt (#2233, #2234, #2235, #2236)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` (two tiers) and
`testdata/tiny-modernbert-bundle`, and three `turbo-kserve-demo` in
front of it: one with `--allow` for every write, one with none, and one
with `--allow` for `CreateIndex` only. Machine: a 4-core x86_64 Linux
container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. Browser tests for
`demo/web/e2e` are follow-up work. `demo/tests/describe.rs` and the unit
tests in `demo/web/test` (`errors.test.ts`, `permit.test.ts`) are in the
repository and run in CI.

- `red.log`: the checks against the page before this change. 17 of 21
  fail.
- `green.log`: the same checks with the change. All 21 pass.

Each paired picture shows before (left) and after (right):

| File | What to look at |
|---|---|
| `no-allow-search.png` | A demo with no `--allow`: New index and Add documents are off, each with the `--allow` that enables it. The Search row and the collection controls have labels. |
| `allow-create-only-search.png` | After only: `--allow` for `CreateIndex` alone. New index is live, the upload is still off. |
| `refused-load-with-details.png` | A refused LoadBundle: a sentence first, the code and raw message behind "details" (opened here). |
| `search-labels.png` | The Search and Collections rows with a label on every control; "no index yet" in the index select. |
| `analyze-form.png` | Analyze with no open-span or relation model served: those options are not shown. |
| `semantic-form.png` | Semantic with no reranker or QA model served: those selects are not shown. |
