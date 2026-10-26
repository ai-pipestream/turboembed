# Integrating turboembed

This page is the path from a document a pipeline already has to chunks,
vectors and entities whose every offset leads back to it. It covers the
two ways in, the server and the library, and walks through one runnable
example: a document in docling's JSON form, analyzed end to end on the
CPU with the small test model, in CI on every change.

## Two ways in

| | The server, `turbo-kserve` | The library, in process |
|---|---|---|
| What it is | gRPC: the Open Inference Protocol for vectors, `turbo.result.AnalysisService` for analysis | The analysis crates under `analysis/`, run by an `Analyzer` |
| Language | Any with gRPC: clients are generated from `server/proto` | Rust (the C interface embeds; see docs/grpc.md for the server's view of it) |
| Structure from a converter (docling) | Not taken: the server analyzes text | Imported as a `Structure` layer, as below |
| Offsets | `OFFSET_MODE_UTF8`, `OFFSET_MODE_UTF16`, `OFFSET_MODE_CODE_POINT` per request | Every view's `OffsetTable`: bytes, UTF-16 units, code points |
| Runs on | The backends linked into the binary | The same |

A pipeline that only needs vectors for text it has already chunked
calls the server. A pipeline that has structure (headings, lists,
footnotes) and wants the chunker to cut along it runs the library, as
the example does.

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
`grpcio-tools` for Python or `protoc-gen-grpc-java` for Java. Published
client packages are tracked in issue #1297; until then, pin the proto
files of the release the server runs.

## The docling example

`examples/docling` is a crate whose test,
`examples/docling/tests/end_to_end.rs`, runs in CI with every workspace
crate. Its input, `examples/docling/data/report.docling.json`, is a
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
(page headers and footers) is not read. Tables, pictures, a list inside
a list and any other label are refused with the item's reference: a
converter that drops content quietly would lose offsets too. Every
item's reference and source span are kept, so any span maps back to the
item and to the offset in its own `text`.

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

### What it does not show

- It runs in process on the CPU backend; the server has no call that
  takes a converter's structure, so a server pipeline sends text and
  chunks it the server's way.
- The server's heatmaps (`InspectDocumentWindow`) are not part of it.
- Tables and pictures from docling are refused rather than converted.
