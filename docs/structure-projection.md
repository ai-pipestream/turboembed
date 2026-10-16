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
