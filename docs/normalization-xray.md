# Normalization x-ray

`turbo_analysis::NormalizationXray` turns an alignment between document
views into contiguous `Equal`, `Replaced`, `Inserted` and `Deleted` runs.
Each side partitions its complete byte domain. Insertions have an empty
parent span; deletions have an empty child span. Empty unchanged gaps are
omitted. Equal runs have identical bytes on both sides.

```rust
use turbo_analysis::{NormalizationXray, ViewId};

# fn inspect(doc: &turbo_analysis::Document, target: ViewId) -> Result<(), turbo_analysis::Error> {
let xray = NormalizationXray::between(doc, ViewId::SOURCE, target)?;
for (index, run) in xray.iter().enumerate() {
    let before = xray.parent_slice(index, doc.source());
    let after = xray.child_slice(index, doc.text(target));
    println!("{:?}: {:?} -> {:?}", run.kind, before, after);
}
# Ok(()) }
```

Both reverse paths and sibling views are supported through their common
ancestor. `steps()` includes the origin followed by the stored views on
the path. Each summary records the actual instance, annotator, version,
configuration hash, original parent, edit count and byte lengths. A
summary's parent always describes that view in the document, even when
the inspection path visits it in reverse.

`from_source_bytes(doc, target)` includes decoding below view zero.
Its parent spans address the original file encoding. Use
`parent_bytes_slice` for that domain; an arbitrary file encoding cannot
be treated as UTF-8 text. The source summary retains the document's
recorded source provenance and includes the decoder's edit count and
byte lengths. It does not invent decoder provenance absent from the
document.

`project_utf16` and `project_code_points` return typed parent and child
span columns in the same run order. Supply each text's own `OffsetTable`,
normally obtained from `Document::offsets`. Domain lengths and character
boundaries are checked before conversion; each side uses a forward
`OffsetCursor`. Projection refuses decoder byte boundaries inside UTF-8
characters. Tables must belong to their supplied text, as required by
the existing offset-table API.

Construction owns offset columns and step metadata; projection owns its
typed output columns. Neither copies a string per run. Iteration,
column access and borrowed text slicing allocate nothing after
construction, verified with an allocation counter. Building the run
columns visits the composed edit list once. Alignment composition and
view-path metadata have their existing construction costs.

Tests cover insertion/deletion/replacement boundaries, empty documents,
reverse and sibling paths, mixed-width Unicode edits, decoder bytes,
typed projection against `OffsetTable`, the actual mojibake/NFC/case
fold/dehyphenation chain and an actual PII detection/masking pipeline.
