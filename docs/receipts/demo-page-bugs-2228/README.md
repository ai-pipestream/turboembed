# Demo page bug fixes receipt (#2228, #2229, #2230, #2231, #2232)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` (two tiers) and
`testdata/tiny-modernbert-bundle`, and two `turbo-kserve-demo` in front
of it: one started with `--allow` for the write methods, one with none.
Machine: a 4-core x86_64 Linux container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository: it clicks through the tabs, stops and starts `turbo-kserve`
under the open page, and prints one line per check. Browser tests for
`demo/web/e2e` are follow-up work.

- `red.log`: the checks against the page as it was on main. 23 of 29
  fail. The three that pass under "header status" pass only because the
  header never left green.
- `green.log`: the same checks with the change. All 29 pass.

| File | What to look at |
|---|---|
| `server-stopped-under-open-page.png` | Before (left) and after (right), the server stopped under an open page: the header dot turns red and a notice offers "Retry now". |
| `opened-with-server-stopped-embed.png` | Before and after, the page opened with the server stopped: the Embed model list says why it is empty. |
| `opened-with-server-stopped-semantic.png` | Before and after: each empty select says the server is not reachable. |
| `confirm-delete-index.png` | After one click on Delete: "Delete demo?" with Yes and Cancel; the index is still there. |
| `refused-delete-beside-row.png` | A delete refused by a demo with no `--allow`: the reason is beside the row. |
| `confirm-unload.png` | After one click on Unload on the Server tab. |
