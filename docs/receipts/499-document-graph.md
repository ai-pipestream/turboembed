# Document graph export boundary

Base: `0dcf0828`.

The boundary lives in `turbo_analysis::export` (core owns the document
it walks). `export` turns a `Document` into a `DocumentGraph`: source
bytes with their decoding alignment, every view with parent and
alignment edits, and one record per layer instance with full layer
name, instance name, schema, status and provenance. Layer bodies cross
only through per-type `LayerCodec`s registered in `Codecs` (shipped:
generic `Edges` and `Groups` in core, `Abbreviations` in the layers
crate), items are named by `NodeRef`, and `DocumentGraph::to_bytes`
is the deterministic canonical encoding at `FORMAT_VERSION` 1.
`import` is strict and atomic, with typed refusals for unknown format
or layer schema versions, missing codecs, duplicate instances,
dangling provenance references and dangling edge or group endpoints,
unknown views, inconsistent source records and malformed bytes.

The round-trip test builds a document from legacy-encoded source
bytes (so source offsets and text offsets differ, around a
supplementary-plane character), adds a transformed view, abbreviation
spans with definition and mention edges, one group, one skipped layer
and one failed layer, then checks the export against the checked-in
fixture `analysis/core/tests/fixtures/document-graph.bin`, imports it
back and verifies source identity and byte mapping, view alignment,
node identity, endpoints, scores, group membership, provenance and
both unresolved statuses, and re-exports to the same bytes. The
`document_graph` example in `turbo-analysis-layers` is the caller
shape a search or coordinator service uses. The versioning and
recovery contract is written in `docs/analysis.md` and in the module
documentation.

Validation, CPU:

```sh
cargo fmt --all --check
cargo test -p turbo-analysis -p turbo-analysis-layers
cargo clippy -p turbo-analysis -p turbo-analysis-layers --all-targets -- -D warnings
cargo run -p turbo-analysis-layers --example document_graph
```

Review focus: strictness choices (no partial import, a codec required
even for skipped and failed layers), the per-import name leak when
rebuilding provenance, the hand-rolled encoding's field rules, and
which layer types still need codecs before a given pipeline's output
can cross.
