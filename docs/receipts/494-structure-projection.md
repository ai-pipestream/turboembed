# Structure across normalization

Base: `272b9298`.

The regression detects source structure around SUMMARY and DETAILS, with
Unicode body text. Detecting structure after whitespace collapse produces
one Paragraph. Projecting source structure preserves all 6 blocks,
hierarchy and source byte mappings.

Validation, CPU:

```sh
cargo fmt --all --check
cargo test -p turbo-structure
cargo clippy -p turbo-structure --all-targets -- -D warnings
```

All passed: 50 tests, no failures or skips. New cases cover named layer
bindings and provenance, source mapping, edits across block edges, erased
blocks, unrelated views, and chunking within projected sections.

This retains available source structure. It cannot recover markup already
missing from an input file. Review focus: direct-parent requirement,
mapping at edit boundaries, rejection of erased blocks, hierarchy and
chunker integration.
