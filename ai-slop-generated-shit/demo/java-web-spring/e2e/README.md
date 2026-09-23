# End-to-end tests for the web demo

Playwright tests that drive `demo/java-web-spring` in a real browser and
exercise both HTTP surfaces directly. The default run starts the app itself on
the committed mock bundles, so it needs no accelerator and no model download,
and every vector, score and generated token is the same on every run.

The Benchmarks panel needs no model: it reads the committed receipts under
`testdata/receipts/turbo/bench`, which `run.sh` points the app at.

The suite starts the app with five models: the mock embedding bundle
(`testdata/bundles/mock/embedding`, through `run.sh`'s default), the mock
generative bundle for the summarizer, and the mock reranker, classifier and
token classifier as `turbo.models[0..2]`. The mock bundles declare a `mock`
tokenizer with no `tokenizer.json`, so the embedding model is given
`testdata/bundles/minilm-tokenizer` as its `turbo.tokenizer-bundle`; the ids
the Tokenize panel shows are that WordPiece vocabulary's.

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

The suite starts its own app on `E2E_PORT` and waits for
`GET /api/v1/health`, so the jar in `demo/java-web-spring/target` must already
exist. To let the suite build it instead, run `TURBO_WEB_SKIP_BUILD=0 npx
playwright test` (that needs Maven).

Environment variables the config reads:

- `E2E_PORT`: the port the suite serves on. Default 8091.
- `E2E_BASE_URL`: test an app that is already running. The suite starts
  nothing and connects to this URL instead.
- `JAVA_HOME`: the JDK 25 to run the app with. Default
  `$HOME/.sdkman/candidates/java/25.0.3-tem`.
- `TURBO_WEB_SKIP_BUILD`: passed to `run.sh`. Default `1`.

## What is covered

`tests/ui.spec.ts`, the Embed panel and the tab bar:

- the device line names the model, the dimension, the sequence and batch
  limits, the device, the provider and ordinal, the runtime version and, when
  the model is not fully accelerated, the stages that ran on the host
- every tab opens its own panel and hides the others, by click and by arrow
  key, Home and End; only the selected tab is in the tab order
- the three sentences the textarea ships with render a 3x3 matrix: 1.000 down
  the diagonal, every cell in [-1, 1], and the matrix symmetric
- the status line carries the batch, the dimension, the server-side time, the
  device, the output placement and the round trip
- `Ctrl` + `Enter` in the textarea runs the embed
- the same sentence entered twice produces identical rows and identical vectors
- an empty textarea shows the server's `texts must not be empty` refusal, the
  page does not throw, and the next embed still works
- one line over the model's batch shows the server's batch refusal naming the
  model and its batch

`tests/rerank.spec.ts`, the Rerank panel: the reranker line, the ranking
ordered best first with every document shown once, the unrelated document
last, input order when the ranking is turned off, a blank query refused with
the server's message and the panel still working afterwards, and the batch
refusal.

`tests/tokenize.spec.ts`, the Tokenize panel: the line naming the bundle the
tokenizer comes from, the pieces with their ids and the two special tokens
marked, `add_special_tokens` off dropping exactly those two, and a second line
tokenized as its own row.

`tests/summarize.spec.ts`, the Summarize panel: the generator line, text
streamed into the output box with the status line ending in the token count,
the rate, the finish reason, the prompt size and the device, the same document
generating the same text twice, an empty document showing the validation
refusal, a prompt over the model's `max_seq` showing the library's
`TURBO_E_CAPACITY` through the mid-stream `error` event rather than an HTTP
status, and Stop leaving the panel usable. The mock model produces its 160
tokens faster than a click, so cancellation itself is covered deterministically
by `GenerateApiTest.aSinkThatStopsCancelsTheGenerationOnTheDevice`, which is
the same path a disconnected client takes.

`tests/devices.spec.ts`, the Devices panel: one card per device with its
feature bits as chips and only its offered capability cells, one card per
loaded model with its contract, labels and prompt prefixes, and the header
links reaching the API explorer, the OpenAPI document and `/v2`.

`tests/benchmarks.spec.ts`, the Benchmarks panel: a summary row and an
expandable cell table for every comparison the endpoint returns, each naming the
receipt file it came from, the floor rule stated once at the top of the panel,
the cells under the floor marked in both the summary and the cell table with a
comparison that has none marking nothing, a comparison expanded to every cell
with both sides' figures and a ratio bar whose width follows the ratio, and the
committed CUDA receipt's 32x32 cell in the per-device throughput table with the
rows and tokens per second the receipt carries. Every expectation is read from
`GET /api/v1/benchmarks`, so a receipt added to
`testdata/receipts/turbo/bench` does not break the suite.

`tests/api.spec.ts`, `/api/v1` directly: health, the whole device survey
including the capability matrix and the bits the mock does not advertise, the
model contracts field by field, embed, similarity, rerank with `top_n` and the
ranking, classify and token-classify with the span offsets checked against the
input bytes, tokenize and detokenize, generation whole and streamed, and every
documented refusal with its status, its `TURBO_E_*` code and its field index.

`tests/oip.spec.ts`, `/v2` as a protocol client: server metadata, the probe
objects, model metadata for each model kind, inference for embedding,
reranker, classifier, token classifier and generative models, `outputs`
narrowing the response, the embedding tensor matching the `/api/v1` vectors
flattened row-major, and the `{"error": "..."}` refusals with 400, 404 and
500.

Every check is a hard assertion. When a call the test expected to succeed is
refused, the failure carries the server's own message.

## Screenshots

`tests/screenshots.spec.ts` writes the images `docs/screenshots` holds. It is
not part of a normal run, because it overwrites committed files:

```bash
cd demo/java-web-spring/e2e && npm run screenshots
```

That writes, at 1200 CSS pixels wide, device scale factor 1, light color
scheme: `docs/screenshots/page.png` (the whole page after embedding the three
defaults), `matrix.png` (the table alone), `rerank.png`, `tokenize.png` and
`devices.png` (each panel after running it), and `benchmarks.png` (the
Benchmarks panel from its heading down to the end of the first throughput
table, with one comparison opened; the whole panel is every committed receipt,
which is far too tall for a README image). The test fails if an image exceeds
400 KB, or 512 KB for `benchmarks.png`, which is a table of every receipt.

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
