# Full text to document graph with the release packages (#1948)

The caller-facing path from text to a document graph, run end to end
with the packaged candidates. Machine B, CPU (AMD Ryzen 9 9950X).

## Packages

| step | package | pin (`analysis/pins/models.txt`) |
|---|---|---|
| `turbo:gliner` | `gliner-community/gliner_small-v2.5` at `f227d3cd637bd4e6757ae143935316d062393341` | package `8a1d791d...fa83` |
| `turbo:zero-shot-relations` | `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e` | tree `34314634...ad38` |

Both are loaded with `LoadBundle` into a running server and named by
`turbo_pins::require`, which refuses a package whose digest is not
pinned. They are local-only and are not shipped.

## What runs

`server/tests/full_text_graph_served.rs`, over a heading and three
sentences in two paragraphs, with diacritics, an emoji and an ideograph
outside the Basic Multilingual Plane, an NFKC ligature, a currency sign,
an email address and a phone number. The steps are normalization,
structure, sentences, tokens, abbreviations, terms, stems, stopwords,
keywords, PII, numbers, money, dates, subwords, GLiNER entities (person,
organization, location at 0.5), GLiNER2 relations (works for, founded,
located in, typed by entity kind, within a sentence, at 0.5), both
boundary layers and the chunker.

- The graph `AnalyzeDocumentProgressive` returns is byte-identical to
  the graph `compile_analyzer_with_models` writes in process from the
  same packages. The same holds for empty text, and for a 366-word
  sentence over GLiNER2's 512 encoder pieces, where the relation step
  fails with `TURBO_E_CAPACITY` in both and every other layer is the
  same.
- Entities and PII slice the source, every token and entity keeps its
  UTF-16 offsets, relation mentions link entity rows, edges are
  deduplicated, and evidence verifies. Each model layer's provenance is
  the digest `Capabilities` reports.
- A second run and `RoundTripDocumentGraph` are byte-identical.
- `InspectDocumentGraph` (UTF-16) gives each entity at the offsets the
  graph maps it to and each relation over the graph's entity rows, and
  each model layer names its package. `InspectDocumentWindow` at 64
  bytes a page pages through the whole source, with every entity on a
  page.
- A model step without its model is refused by name in process.

The browser side, run by `scripts/e2e-web.sh` (`demo/web/e2e/install.test.ts`
and `demo/web/e2e/acceptance_isolation.test.ts`): a user sees a model's
terms still to accept, accepts the exact text, installs the model, every
file verifies against the catalog, and the model is loaded into the
running server and used, with no restart. A second user against the same
server keeps isolated acceptances and cannot load the share-alike
bundle before accepting; a browser sending a spoofed user name in a
binary protobuf request has its user field overwritten by the demo proxy;
an anonymous demo without `--user` is read-only (403 on direct writes);
and a token-protected demo enforces the bearer token (401 without it)
and `--allow` write controls. That run uses the tiny test bundle under
share-alike terms from a local catalog, not these packages.

## Result

```text
GLINER_PACKAGE=gliner-small-v2.5-package GLINER2_PACKAGE=gliner2-package \
  cargo test --release -p turbo-kserve --test full_text_graph_served -- --include-ignored
  # 4 passed, 9.75 s
scripts/e2e-web.sh
  # 70 passed, 0 skipped
```

## Not covered

- Gazetteer places and GLiNER entities in one request: both write
  `Entities`, so the served run leaves out `turbo-geo:names`. The
  in-process model-free run (`server/tests/full_text_graph.rs`) has it.
- The release packages installed through the catalog: the browser run
  installs a test bundle.
