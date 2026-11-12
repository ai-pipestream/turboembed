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

## Two fixes after review (`review/`)

Two faults found in review of this change, each with its own driving
script, run against the page before the fix (red) and after (green).

- A page opened while the server was live and still loading stayed
  empty after the server turned ready. The check starts `turbo-kserve`
  with 60 names of `testdata/tiny-bert-bundle` at 8 sessions each, which
  takes about ten seconds to load, opens the page at once and waits for
  the header to say ready. `ready-red.log`: 2 of 4 pass, the Analyze
  model list and the Server tab stay empty. `ready-green.log`: all 4
  pass. `loading.png` and `ready.png` are the same page a few seconds
  apart, with no reload between them.
- Vocabulary's Import, Learn and Download shared one cancel signal and
  one Stop button, so starting a second ended the first and turned Stop
  off under the call still running. The check uploads a 4 MB file to
  Learn and then to Import over a link held to 48 KiB a second.
  `stop-red.log`: 4 of 7 pass. `stop-green.log`: all 7 pass: both run
  side by side, Stop stays on, and one press ends both.
  `two-running.png` shows both buttons counting with Stop on.

## Every tab swept after a second review (`review/sweep-*`)

A second review found the same two patterns in places the first fix had
not reached, so every tab was checked for them with one driving script,
run against the page before (`sweep-red.log`, 8 of 24 pass) and after
(`sweep-green.log`, all 24 pass).

- The page now keeps one status, up and ready, and every tab that reads
  from the server reads again on the same events: the first answer, the
  server going away, coming back, and turning ready. Before, the Models
  tab's teachers, the Vocabulary selects, the Search rerank select and
  the model sources were read once when the page opened and never
  again, and showed a raw connection error or stayed on "Loading".
  `sweep-stopped-models.png` and `sweep-started-models.png` are the
  Models tab of one page, opened with the server stopped and then a few
  seconds after it started, with no reload.
- No select is blank: one with nothing to offer says why (the server is
  not reachable, it serves none of that kind, or no document has been
  analyzed yet).
- A demo that refuses the call for want of its token is up, not
  unreachable: the notice under the tabs stays hidden and the tabs fill
  once the token is typed.
- Escape is ignored by a confirmation while its Yes is running. Before,
  it brought the first button back over the call in flight and a second
  Yes sent the delete again (`sweep-red.log`: 2 calls). The check holds
  the link at 1.5 seconds a call and sends the key to the confirmation.
- Install, Reinstall and Save on the Models tab, Accept on a licence,
  Add on the Embed tab and Connect Google Drive are off and show the
  elapsed time while their calls run, like every other button
  (`sweep-install-busy.png`).
