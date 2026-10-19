### Structure across normalization

Detect or import `Structure` before changing layout. Bind that layer as
`source`, normalize into view `clean`, then add:

```rust,ignore
.step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
```

`turbo_structure::StructureProjector` maps blocks from the direct parent
view to the normalized view and preserves kinds, parents and depths. Keep
the original layer for source evidence. Project each step when normalization
uses a chain of views. Unrelated views, edits across block edges and erased
blocks return errors. This cannot infer layout missing from the input.

One import source is a MASC document's logical markup: read the corpus
with `MascCorpus::read_with_logical` (or one document with
`read_document_with_logical`), take `MascDocument::structure()` for the
`Structure` over the source text, and add it to an analysis document with
`Provenance::by("masc:logical")` bound as `source`. The document region
(`cesDoc`) becomes the root section, `head` blocks become headings and
`p` blocks paragraphs; every other label stays available unconverted in
`MascDocument::logical()`.
