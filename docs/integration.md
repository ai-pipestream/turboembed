# Integrating turboembed

This page is the path from a document a pipeline already has to chunks,
vectors and entities whose every offset leads back to it. It covers the
two ways in, the server and the library, and walks through one runnable
example: a document in docling's JSON form, analyzed end to end on the
CPU with the small test model, in process and against a running server,
in CI on every change.

## Two ways in

| | The server, `turbo-kserve` | The library, in process |
|---|---|---|
| What it is | gRPC: the Open Inference Protocol for vectors, `turbo.result.AnalysisService` for analysis | The analysis crates under `analysis/`, run by an `Analyzer` |
| Language | Any with gRPC: clients are generated from `server/proto` | Rust (the C interface embeds; see docs/grpc.md for the server's view of it) |
| Structure from a converter (docling) | Imported by the client into a seed graph, sent as `initial_graph` to `AnalyzeDocumentProgressive` | Imported as a `Structure` layer, as below |
| Offsets | `OFFSET_MODE_UTF8`, `OFFSET_MODE_UTF16`, `OFFSET_MODE_CODE_POINT` per request | Every view's `OffsetTable`: bytes, UTF-16 units, code points |
| Runs on | The backends linked into the binary | The same |

A pipeline that only needs vectors for text it has already chunked
calls `AnalyzeStream`. A pipeline that has structure (headings, lists,
footnotes) and wants the chunker to cut along it either sends that
structure to the server's pipeline call or runs the library; the example
does both.

## Running the server

Getting the binary, its flags and how it stops are in
[grpc.md](grpc.md); release archives and what each carries are in
[release.md](release.md), which also holds the versioning and
compatibility policy (what a minor or patch release may change on the
wire and in a bundle). Container images are tracked in issue #1298.

```
turbo-kserve --listen 0.0.0.0:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
```

`testdata/tiny-bert-bundle/manifest.json` is the small test model the
example uses: `sentence-transformers/all-MiniLM-L6-v2` shrunk to 32
dimensions and 64 tokens, with the same tokenizer. Real bundles, and how
to make one, are in [bundle.md](bundle.md) and [zoo.md](zoo.md).

### Health

The server serves `grpc.health.v1.Health` (`Check` and `Watch`):
`NOT_SERVING` while bundles load and while it drains, `SERVING` once every
model's sessions exist, which is when the Open Inference Protocol's
`ServerReady` turns true ([grpc.md](grpc.md)). A Kubernetes probe or a
load balancer asks it:

```
grpcurl -plaintext localhost:8081 grpc.health.v1.Health/Check
```

### The calls an integration makes

| Need | Call | Where |
|---|---|---|
| What is loaded, and every limit | `AnalysisService/Capabilities` | [grpc-analysis.md](grpc-analysis.md) |
| Vectors for one batch | `GRPCInferenceService/ModelInfer` | [kserve.md](kserve.md) |
| Vectors, sparse terms or token vectors for many documents | `AnalysisService/AnalyzeStream` | [grpc-analysis.md](grpc-analysis.md) |
| One document through a named pipeline, seeded with a converter's structure | `AnalyzeDocumentProgressive` | [The docling example on the server](#on-the-server) |
| Inspecting one document: segments, heatmaps, chunk projections | `InspectDocumentWindow`, `ProjectChunks` | [grpc-inspect.md](grpc-inspect.md) |
| Exporting an analyzed document | `ListOutputFormats`, `FormatDocument` | [grpc-inspect.md](grpc-inspect.md) |
| Loading or unloading a bundle at run time (`--admin`) | `LoadBundle`, `UnloadBundle` | [grpc-analysis.md](grpc-analysis.md) |

Read `Capabilities` first: `context_limit` and `batch_limit` per bundle,
the stream caps under `server.stream`, and the result-message limits
under `limits`. A request past one is refused, never truncated.

### Offsets

A span crosses the wire in the mode the request names. Java, JavaScript
and C# strings index UTF-16 units: ask for `OFFSET_MODE_UTF16` and index
the string directly. Python strings index code points:
`OFFSET_MODE_CODE_POINT`. `OFFSET_MODE_UTF8` is the raw byte offset into
the UTF-8 text sent. All three name the same span; the server converts
with the text's own offset table.

### Errors

Every refusal is a gRPC status with two trailing metadata entries:
`turbo-code`, the library's status code in decimal, and `turbo-field`,
the number of the field that was refused (0 when none). The table, and
which gRPC status each code maps to, is in [kserve.md](kserve.md),
Errors. A client retries `UNAVAILABLE` (no idle session) and treats
`INVALID_ARGUMENT` and `OUT_OF_RANGE` (`TURBO_E_CAPACITY`, a limit named
in the message) as the request's fault.

### Clients

Java, Python, Go and other clients are generated from the protos in
`server/proto` with each language's gRPC tooling, for example
`grpcio-tools` for Python or `protoc-gen-grpc-java` for Java. The
release workflow packages them as a Maven artifact (`clients/java`) and
a wheel (`clients/python`), built by `scripts/package-proto-stubs.sh`;
publishing them is tracked in issue #1297. Until then, build them with
that script or pin the proto files of the release the server runs. The
docling example has a Java and a Python client written against those
stubs, below.

## The docling example

`examples/docling` is a crate whose tests,
`examples/docling/tests/end_to_end.rs` (in process) and
`examples/docling/tests/server.rs` (against the server), run in CI with
every workspace crate. Its input, `examples/docling/data/report.docling.json`, is a
short report as docling exports it: a title, numbered section headers at
two levels, paragraphs with doubled spaces and line breaks, a list of
three items, a footnote, and a page header.

```
cargo test -p turbo-example-docling
```

### From docling's JSON to a document

`turbo_example_docling::docling::parse` walks the body in reading order
and lays each text item's `text` down once, blocks a blank line apart
and list items one line apart. That string is the source text: nothing
is joined or copied after it. Each item becomes a block of a
`Structure` layer over it, added under the name `source`:

| docling | Block | Depth |
|---|---|---|
| `title` | `Heading`, opening a `Section` | 1 |
| `section_header` | `Heading`, opening a `Section` | `level` + 1 |
| `text`, `paragraph`, `reference` | `Paragraph` | 0 |
| `list` group | `List` | 1 |
| `list_item` | `ListItem`, in its list | 1 |
| `caption`, `footnote`, `code` | `Caption`, `Footnote`, `Code` | 0 |

A section runs to the next heading at its depth or above. Furniture
(page headers and footers) is not read. Tables, pictures, text items with children, a list inside
a list and unsupported text labels are refused with the item's reference: a
converter that drops content quietly would lose offsets too. Every
item's reference and source span are kept, so any span maps back to the
item and to the offset in its own `text`. Malformed body/children fields,
out-of-range heading levels, repeated/cyclic references and nesting past
128 references return errors. Unsupported descendants are not discarded.

### The pipeline

`turbo_example_docling::pipeline` builds one `Analyzer`:

1. `NormalizeStep` with `Whitespace::keep_paragraphs()` writes the
   `clean` view: the blank line between blocks becomes one line break,
   and every other run of whitespace (doubled spaces, a line break inside
   a paragraph, the break between list items) one space. Every edit is
   kept in the view's alignment.
2. `StructureProjector` carries the docling structure from the source
   onto `clean`, refusing an edit that crosses a block's edge
   ([structure-projection.md](structure-projection.md)).
3. `RuleSentenceDetector` and `SimpleTokenizer` on `clean`.
4. `SubwordAnnotator`: the model's own WordPiece subwords, once. The
   chunker's sizes and the vectors are counted in these.
5. `DictionaryNameFinder` marks the places named in the dictionary
   (`location`).
6. `SentenceVectors`: a vector per sentence from the bundle.
7. `StructureBoundaries` (a cost per gap from the blocks),
   `DriftBoundaries` (a cost per gap from how far the sentence vectors on
   either side drift apart, two sentences a side) and `WeighBoundaries`
   (the two, weighed into one).
8. `Chunker` with a budget of 48 subwords, aiming for 32, and
   `ChunkVectors`: a vector per chunk, at every level.

How each step is configured, and the others it could use, is in
[analysis.md](analysis.md).

### What it checks

`report::check_offsets` takes every token, subword, entity and chunk of
`clean`, maps it to source bytes and back to the same span, finds the
docling items it covers, and for everything but chunks compares the item's
own text at those offsets with the clean text, byte for byte. A
normalizing edit, a projection or a separator that broke the mapping
fails it.

`report::report` writes the result as JSON: per chunk its level, parent,
cut, size in subwords, whether it is over budget, its span in `clean`
(bytes, UTF-16 units and code points), in the source, and per docling
item (bytes, UTF-16 units and code points within that item's `text`),
and the first values of its vector; per entity the same spans. The test
holds it to `examples/docling/data/report.expected.json`: spans, levels
and entities exactly, vector values within 0.001. A change that moves a
chunk shows as a diff of that file; `BLESS=1` rewrites it to be
reviewed.

An entity in the report reads, for example: the `Zürich` at clean bytes
105 to 112 (UTF-16 105 to 111) is source bytes 108 to 115, since the
normalizer removed a doubled space before it, and bytes 70 to 77 of
`#/texts/3`.

### On the server

The same document goes through a running `turbo-kserve` on the CPU.
Only the step a converter's output needs stays on the client: reading
docling's JSON into the source text and a `Structure` layer, the
default instance, over it. That one-layer document is exported as
DocumentGraph bytes (`turbo_example_docling::server::seed`, checked in as
`examples/docling/data/report.seed.graph`) and sent as `initial_graph`
to `AnalyzeDocumentProgressive` with these steps and `model`
`tiny-bert-bundle`:

```
turbo:normalize:whitespace     turbo:structure-boundaries
turbo-sentdetect:rules         turbo:chunker
turbo-tokenize:uax29-untyped   turbo:levels
turbo:subwords                 turbo:chunk-vectors
turbo-geo:names
```

The server keeps the structure as given, runs sentences, tokens,
subwords and names on the normalized view and projects them back, so
every span in its answer is in the source text docling's items were laid
down in. The stream's last event carries the analyzed graph. The client
sends it back twice:

- `InspectDocumentWindow` with `heatmap_buckets` 16: the heat, a token
  density and an entity density lane, 16 buckets each, plus the window's
  text and every segment's layers (chunks, entities, structure).
- `ProjectChunks`: a point per chunk with its level and span, and
  whether it has a vector.

Three clients do this and print the same summary, held to
`examples/docling/data/report.server.expected.txt`:

| Client | Where | Offsets asked for | Runs in CI |
|---|---|---|---|
| Rust | `examples/docling/src/bin/docling-client.rs` | `OFFSET_MODE_UTF8` | `examples/docling/tests/server.rs`, a server started in process on a free port, every call over gRPC |
| Java | `examples/docling/clients/java/DoclingClient.java` | `OFFSET_MODE_UTF16` | the release workflow's `proto-and-stubs` job, against the release binary |
| Python | `examples/docling/clients/python/docling_client.py` | `OFFSET_MODE_CODE_POINT` | the same job, on the oldest Python the wheel declares |

Each slices the window's text by the offsets it asked for, in its own
string type, so the three agree only if every offset mode is right. The
Rust client also decodes the answered graph into a fuller report, held
to `examples/docling/data/report.server.expected.json` (vector values
and heat scores within 0.001): per chunk its level, cut, size and the
head of its vector, and per chunk and entity its spans in bytes, UTF-16
units and code points and the docling items it covers.

Against a server of your own:

```
turbo-kserve --listen 127.0.0.1:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
cargo run -p turbo-example-docling --bin docling-client -- \
    run http://127.0.0.1:8081 examples/docling/data/report.docling.json
python3 examples/docling/clients/python/docling_client.py 127.0.0.1:8081 \
    examples/docling/data/report.seed.graph \
    examples/docling/data/report.server.expected.txt
```

`--report` prints the Rust client's full report instead of the summary;
`docling-client seed DOCLING_JSON OUT` writes the seed for another
document. The Java client compiles against the stub jar and runs the same
way (`java ai.pipestream.turboembed.examples.DoclingClient ADDR SEED
[EXPECTED]`). The summary starts:

```
steps 9
chunk 0 level 1 vector yes "Grid Storage Review 2026\n\n1. Summary ..."
chunk 1 level 0 vector yes "Grid Storage Review 2026\n\n1. Summary\n\n..."
...
entity 0 location "Zürich"
...
heat tokens TokenDensity 6 5 7 7 6 6 5 6 5 6 7 7 6 8 6 6
heat entities EntityDensity 0 0 0 1 0 1 1 1 0 0 0 2 0 0 0 0
```

`device=0` is the CPU: the server's default device choice never picks
it. `BLESS=1 cargo test -p turbo-example-docling` rewrites the seed and
both expected files, to be reviewed.

### What it does not show

- The server pipeline is the server's steps, not the in-process one:
  no drift boundaries, the bundled gazetteer for place names instead of
  the example's dictionary, and a chunk budget from the model's width
  (64 subwords) instead of 48.
- The Java and Python clients run where the stub packages are built,
  the release workflow, not in the per-change `cargo test` run.
- Tables and pictures from docling are refused rather than converted.
