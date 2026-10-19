# MASC logical headings and paragraphs as Structure

Base: `0dcf0828`.

The header's `f.logical` file (`*-logical.xml`) is now read under
`turbo_formats::masc`: `MascDocument::add_logical` resolves regions,
nodes and annotations by id (the files list them shuffled) and keeps
each labelled stretch in UTF-16 units, and `MascDocument::structure`
converts the `cesDoc`, `head` and `p` blocks to a `Structure` layer
(section, headings, paragraphs, nested by containment) with UTF-8 byte
spans through the existing UTF-16 conversion. Empty regions are kept
in `logical()` and skipped by the conversion; crossing blocks and
unresolvable references are typed errors. The regression fixtures
cover the tree, shuffled references, UTF-16 anchors around a surrogate
pair, anchors inside a pair or past the text, malformed references and
projection to a normalized view with `masc:logical` provenance.

Validation, CPU:

```sh
cargo fmt --all --check
cargo test -p turbo-formats
cargo clippy -p turbo-formats --all-targets -- -D warnings
TURBO_TEST_MASC=/path/to/MASC-3.0.0/data cargo test -p turbo-formats --test masc_logical
```

With a local MASC 3.0.0 checkout the corpus test verifies IntroDublin
imports as 1 section, 5 headings and 15 paragraphs, every span inside
the source text. A whole-corpus run over all 392 documents with
logical files converted 387 cleanly; 4 documents carry an anchor one
unit past the end of their text and 1 has a paragraph crossing its
section, and those 5 fail with the typed errors above. That run is a
local smoke check, not a checked-in test, and CI needs no corpus path.

Review focus: reference resolution and its refusals, the label to
block mapping (only `cesDoc`, `head` and `p` convert), empty-region
handling, and the opt-in corpus entry points beside the chunk files.
