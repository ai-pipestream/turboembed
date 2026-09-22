# End-to-end tests for the web demo

Playwright tests that drive `demo/java-web-spring` in a real browser and check
the JSON API directly. The default run starts the app itself on the committed
mock bundle (`testdata/bundles/mock/embedding`), so it needs no accelerator and
no model download, and the vectors are the same on every run.

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
--server.port=8091` and waits for `GET /api/info`, so the jar in
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

`tests/api.spec.ts`, against the API directly: the fields and values of
`GET /api/info`, the shape of a `POST /api/embed` response (unit vectors, a
symmetric cosine matrix, the trimmed texts echoed back), determinism across two
identical requests, and the two 400 refusals with their messages.

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

`docs/screenshots/page-minilm.png` comes from a run on a real model, so it
needs an app already serving one:

```bash
cd demo/java-web-spring/e2e
SHOT_TARGET=real E2E_BASE_URL=http://127.0.0.1:8092 npm run screenshots
```

The test refuses to write that file if the page reports the mock model.
