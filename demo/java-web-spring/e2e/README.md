# End-to-end tests for the web demo

Playwright tests that drive `demo/java-web-spring` in a real browser and check
the JSON API directly. The default run starts the app itself on the committed
mock bundles (`testdata/bundles/mock/embedding` for the vectors and
`testdata/bundles/mock/generative` for the summarizer), so it needs no
accelerator and no model download, and both the vectors and the generated
tokens are the same on every run.

## Running

From the repository root:

```bash
demo/java-web-spring/run.sh          # builds the jar and serves the demo; stop it with ctrl-c
cd demo/java-web-spring/e2e && npm ci && npx playwright test
```

`npm ci` installs Playwright; the browser itself is a separate download:

```bash
npx playwright install chromium
```

The suite starts the app with `TURBO_WEB_SKIP_BUILD=1 ../run.sh
--server.port=8091 --turbo.generate-bundle=<testdata/bundles/mock/generative>`
and waits for `GET /api/info`, so the jar in
`demo/java-web-spring/target` must already exist. To let the suite build it
instead, run `TURBO_WEB_SKIP_BUILD=0 npx playwright test` (that needs Maven).

Environment variables the config reads:

- `E2E_PORT`: the port the suite serves on. Default 8091.
- `E2E_BASE_URL`: test an app that is already running. The suite starts
  nothing and connects to this URL instead.
- `JAVA_HOME`: the JDK 25 to run the app with. Default
  `$HOME/.sdkman/candidates/java/25.0.3-tem`.
- `TURBO_WEB_SKIP_BUILD`: passed to `run.sh`. Default `1`.

## What is covered

`tests/ui.spec.ts`, through the page:

- the device line names the provider, the device and the model from `/api/info`
- the three sentences the textarea ships with render a 3x3 matrix: 1.000 down
  the diagonal, every cell in [-1, 1], and the matrix symmetric
- the same sentence entered twice produces identical rows and identical vectors
- an empty textarea shows the server's `no texts` refusal in the error box, the
  page does not throw, and the next embed still works
- 17 lines shows the server's batch refusal naming the server's batch

`tests/summarize.spec.ts`, through the page:

- the generator line names the model, the device and the sequence limit from
  the `generate` object in `/api/info`
- Summarize streams generated text into the output box and the status line ends
  with the token count, the rate, the finish reason and the prompt size; the
  tokens the status counts are the tokens the page shows
- the same document generates the same text twice
- an empty document shows the server's `no text` refusal, the page does not
  throw, and the next run still works
- a document whose prompt is over the generative model's `max_seq` shows the
  library's `TURBO_E_CAPACITY` refusal, which is the mid-stream `error` event
  path rather than an HTTP status

`tests/api.spec.ts`, against the API directly: the fields and values of
`GET /api/info` including its `generate` object, the shape of a
`POST /api/embed` response (unit vectors, a symmetric cosine matrix, the
trimmed texts echoed back), determinism across two identical requests, the
`POST /api/summarize` event stream (every `chunk` before the single `done`, the
per-step token counter, `finish LENGTH` at `maxNewTokens`, the prompt size),
and the 400 refusals with their messages.

Not covered: the 409 a summarize call returns when no generative bundle is
configured, which needs a second server started without
`--turbo.generate-bundle`.

Every check is a hard assertion. When an embed the test expected to succeed is
refused, the failure carries the server's own message.

## Screenshots

`tests/screenshots.spec.ts` writes the images `docs/screenshots` holds. It is
not part of a normal run, because it overwrites committed files:

```bash
cd demo/java-web-spring/e2e && npm run screenshots
```

That writes `docs/screenshots/page.png` (the whole page after embedding the
three defaults) and `docs/screenshots/matrix.png` (the table alone), at 1200
CSS pixels wide, device scale factor 1, light color scheme. The test fails if
an image exceeds 400 KB.

The other two images come from runs on real models, so each needs an app
already serving one:

```bash
cd demo/java-web-spring/e2e
# docs/screenshots/page-minilm.png, from an app serving a real embedding model
SHOT_TARGET=real E2E_BASE_URL=http://127.0.0.1:8092 npm run screenshots
# docs/screenshots/summary-qwen.png, from an app serving a real generative model
SHOT_TARGET=generate E2E_BASE_URL=http://127.0.0.1:8094 npm run screenshots
```

Each of those tests refuses to write its file if the page reports the mock
model, and prints the device line, the status line and the generated text it
captured.
