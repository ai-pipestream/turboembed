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

## Table-aware chunks and header context

`turbo_chunker::TableBoundaryAnnotator` reads the view, sentences, tokens
and `Tables`. It writes `Boundaries`: a high positive cost inside a
logical row and a lower cost between rows. The optimizer minimizes costs.
Both configured costs participate in its deterministic configuration hash;
inside-row cost must exceed row-edge cost. Token endpoints must be in
source order.

The base `Chunker` now has an optional sixth read, `Tables`. The existing
sentence, subword, boundary, structure and clause slots retain their
positions. When tables are present, each logical row is an atomic unit,
including the header's delimiter or a multiline ASCII row. An oversized
row stays whole and is marked `Chunks::over_budget`, even when clause
splitting is enabled. Sentence spans and top-level section cuts must align
with rows; a crossing returns an error naming the conflict. A newline
sentence detector supplies suitable spans for the supported grid formats.
Chunk spans include complete row envelopes, and model-token counts include
subwords overlapping those source envelopes, including line breaks.
Enclosing structure is tested against that complete token count before
its hard limits are chosen. Chunker provenance uses version 2.

The incremental `Chunker::rechunk` API has no `Tables` argument. Its
full-chunking equality guarantee concerns the inputs it accepts, without
table-row atomicity. After a table edit, use a full Chunker run with the
updated tables, followed by TableHeaderAnnotator to replace header context.

After chunking, `turbo_chunker::TableHeaderAnnotator` reads the view,
`Chunks` and `Tables`, writing `ChunkTableHeaders`
(`turbo:chunk-table-headers`). `for_chunk(chunk_ix)` returns all contexts
for that chunk, ordered by table ID. Each context has the chunk and table
IDs, original header cell envelopes, a Markdown header/alignment prefix,
and `is_continuation`. The flag is true when the chunk lacks any original
header fragment. The prefix and coordinates are available on every chunk;
consumers prepend the prefix only when continuation is true. Headerless
grids have empty header fields and a false continuation flag.

Multiline headers render their actual cell fragments, separated by spaces.
Multiple header rows render one label per governed column, joined with
` / `. Literal pipes are escaped without doubling existing source escapes.
A per-table index collects and sorts only header cells once per layer;
it never collects or sorts body cells per chunk.

The prefix is context metadata. Consumers that prepend it before embedding
must reserve its model-token cost in their chunk budget. `Chunker` counts
source spans plus its configured overhead; it does not silently add the
rendered header to either source text or token counts.

The standard layer codecs register `ChunkTableHeaders` too. Its primary
spans copy each chunk's fragments. Validation resolves the actual named
chunk/table dependencies and checks IDs, exact source spans, prefix,
continuation and every overlapping chunk/table pair. Secondary header
coordinates may be outside a continuation chunk, but must match the
original table and the exact view's UTF-8 boundaries. These checks also
run on binary graph import.

`cargo test --locked -p turbo-chunker --test table_header --test table_header_named --test table_header_refusals --test table_header_allocations`
exercises the production detector, tokenizer, bundle tokenizer, chunker,
header annotators and graph codecs. The 30-row Markdown case has exactly
four chunks, preserves every logical row and carries the original headers
on all four chunks. The allocation regression checks that the header stage
allocates a bounded amount when body rows grow from 30 to 3,000, with the
same four chunks; it makes no timing claim.
