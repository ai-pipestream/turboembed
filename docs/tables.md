# Source-coordinate tables and key/value fields

`TableGridAnnotator` reads a `View` and writes `Tables` (`turbo:tables`).
`KeyValueAnnotator` reads a `View` and writes `KeyValuePairs`
(`turbo:key-value-pairs`). Both run as ordinary bound analyzer steps,
and use the existing block detector to exclude fenced and indented code.

```rust
use turbo_analysis::{Analyzer, Bound};
use turbo_structure::{KeyValueAnnotator, TableGridAnnotator};

let analyzer = Analyzer::builder()
    .step(Bound::new(TableGridAnnotator::default()))
    .step(Bound::new(KeyValueAnnotator::default()))
    .build()?;
```

## Table coordinates

The base library's `Span` holds byte coordinates; `Spans<L>` holds
coordinate columns for layer `L`. `Tables::spans` and `Tables::cells`
therefore form parallel columns. Each `TableCell` names its table,
logical row and column, row/column extent, header flag and alignment.
`cells_in_table` yields borrowed metadata and source envelopes in
row-major order. `headers_for_table` selects its header cells.
`governing_headers` returns the nearest preceding column header and the
column-zero cell of the same row, the row label. These convenience queries
collect and sort the table cells on each call: O(C log C) time and O(C)
scratch for C cells. Cache their results for repeated cell queries;
calling governing_headers for every cell costs O(C^2 log C).

`table_bounds` holds each complete table. `rows`, `row_table` and `row_ix`
hold logical row bounds and coordinates. Rows partition the full source
table, including syntax and line breaks: the GFM delimiter belongs to
the header row, and ASCII borders belong to the preceding row. The
opening ASCII border belongs to the first row. Consumers can preserve
whole rows without re-parsing cell text.

Pipe-cell boundaries follow the [GFM table extension](https://github.github.com/gfm/#tables-extension-):
leading/trailing pipes are optional, escaped pipes stay in cells,
unescaped pipes delimit cells even inside backticks, short body rows
receive empty cells, and excess body cells are ignored. Header and
delimiter column counts must agree. Alignment comes from delimiter
colons. Cell spans preserve source bytes, including escape characters,
and exclude padding; an empty cell is an exact zero-length source span.

ASCII `+---+` grids use a `+===+` header separator to identify header
rows. Grids without that separator have no inferred header. Consecutive
physical content lines form one logical row; multiline cell content uses
the standard discontinuous-span fragments. Parsed cell extents are one
row and one column; merged grid cells are not inferred. A content row
with a different column count ends the grid at its last closed row. A
later border may start another grid; unsupported spanning-cell rows
are excluded rather than treated as rectangular cells.

The default cell limit is 1,000,000 per view, including implied empty
cells. `TableGridAnnotator::max_cells` configures it and participates in
the configuration hash. Exceeding it or the `u16` row/column coordinate
range returns an error, with no partial grid.

## Key/value fields and graph transport

Candidate fields use `:` or `=` and a key containing letters, at most
`max_key_tokens` Unicode-whitespace-separated tokens (six by default).
URLs, numeric timestamp keys and longer prose keys are excluded. Keys,
values and separators keep their exact source coordinates; values may
contain URLs and punctuation. Code blocks are excluded.

`turbo_analysis_layers::export::register_codecs` registers both layers.
The existing `DocumentGraph` binary boundary carries their typed
columnar payload, view identities and provenance. It has no generic
`LayerData` union; no new core graph representation is introduced.

Layer validation checks every cell, table/row bound, key, value and
separator against the exact view and UTF-8 boundaries. It also checks
coordinate ordering, internal table/row IDs, rectangular cell tiling,
positive extents, and row coverage. Secondary coordinates use the
existing `Layer::check_document` hook, including on graph import.

`cargo test --locked -p turbo-structure --test tables --test table_syntax --test table_refusals`
exercises both parsers, the repository analysis document, analyzer
provenance, graph bytes and malformed secondary-coordinate records.
