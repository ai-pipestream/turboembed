# Demo page convenience receipt (#2241, #2243, #2244, #2245)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` (three names) and
`testdata/tiny-modernbert-bundle`, and `turbo-kserve-demo` in front of
it with `--allow` for every write. A second server serves one bundle,
which the check unloads through the page, behind a demo that allows
`UnloadBundle`. Machine: a 4-core x86_64 Linux container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. The uploads are 3 MiB files
with the browser's upload rate held to 1 MiB a second (64 KiB a second
for the one that is stopped), so the bar can be seen moving on one
machine. Browser tests for `demo/web/e2e` are follow-up work. The unit
tests in `demo/web/test` (`keep.test.ts`, `csv.test.ts`,
`about.test.ts`, `dom.test.ts`) and the tree test's rule for the upload
module are in the repository and run in CI.

- `red.log`: the checks against the page before this change. 24 of 29
  fail. Of the five that pass, two pass for the wrong reason there: a
  reload loses everything typed, so the sample is "back" without a
  reset.
- `green.log`: the same checks with the change. All 29 pass.

Each paired picture shows before (left) and after (right):

| File | What to look at |
|---|---|
| `upload-progress.png` | After only: learning a vocabulary from a 3 MiB file shows a bar and the bytes sent of the total, with Stop on. |
| `search-upload-stop.png` | Adding documents to an index: the upload has a Stop and a bar. |
| `kept-after-reload.png` | Embed after a reload: the typed texts, the picked role and the unpicked model are back; "Reset to sample" is under the form. |
| `take-controls.png` | Copy and Download CSV under the similarity matrix and under the vectors. |
| `needs-hint.png` | After the only model is unloaded: the tab's description, and what it needs and where to get it. |
