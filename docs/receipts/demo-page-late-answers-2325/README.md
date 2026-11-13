# Demo page late answers receipt (#2325)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` (served at three precisions) and
`testdata/tiny-modernbert-bundle`, started with `--admin`, and
`turbo-kserve-demo` in front of it with `--allow` for every write.
Machine: a 4-core x86_64 Linux container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. It holds the answer to a call
(the call is made at once, its real answer is handed to the page later),
makes a newer call of the same kind, and then releases the held answer.
The same two cases are in the repository as `demo/web/e2e/late.test.ts`,
and the counter itself is tested in `demo/web/test/latest.test.ts`.

- `red.log`: the checks against the page before this change. 4 of 10
  fail: the held read of what is served brings an unloaded model back to
  the Server table and to the model lists, and the held document window
  of the first layer pick replaces the text and the legend of the second.
- `green.log`: the same checks with the change. All 10 pass.
