# Demo page results receipt (#2237, #2238, #2239, #2242)

The page in Chromium through Playwright, against `turbo-kserve` on the
CPU with `testdata/tiny-bert-bundle` (two tiers) and
`testdata/tiny-modernbert-bundle`, and `turbo-kserve-demo` in front of
it with `--allow` for every write. Machine: a 4-core x86_64 Linux
container, no GPU.

The checks were driven by a Playwright script that is not part of the
repository; it prints one line per check. Browser tests for
`demo/web/e2e` are follow-up work. The unit tests in `demo/web/test`
(`heat.test.ts`, `marks.test.ts`, `highlight.test.ts`) are in the
repository and run in CI.

- `red.log`: the checks against the page before this change. 17 of 18
  fail. It was taken before the layer chips of #2240 landed on their
  own: with them its two "layer chips" lines pass on the old page too.
  This change keeps those chips and their `[instance]` label, and keys
  the legend and the colours by the same layer and instance.
- `green.log`: the same checks with the change. All 18 pass.

Each paired picture shows before (left) and after (right):

| File | What to look at |
|---|---|
| `side-by-side.png` | Embed with two models: the spread column in fixed decimals, the last row label on two lines instead of cut off. |
| `similarity-own-range.png` | After only: cosines from 0.892 to 0.968 coloured by their own range, each row's best match outlined, and the switch back to the absolute scale. |
| `annotated-sentences.png` | Analyze with one layer picked: a legend, and the list of annotations on the stretch that was clicked. |
| `annotated-two-layers.png` | Two layers picked: one fills, the next underlines, each in its legend colour. |
| `layer-chips.png` | After only: the `turbo:boundaries [lexical]` chip picked, with the legend naming that instance. |
| `search-hits.png` | A hit with its matched words marked in place, a legend of the signals, and a scale on the document heat. |
| `phone-analyze-after-run.png` | At 390 px wide, after Run: before, the form is still on screen; after, the results are. |
