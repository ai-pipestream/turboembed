# Structure block conformance

Issue #948 adds raw HTML blocks, footnote definitions, list tightness and
aligned-column plain-text tables. No runtime dependency was added.
`StructureAnnotator::VERSION` is 1. New `LooseList` and `Html` block tags
are appended after the 14 existing tags in both graph and server codecs;
existing tag numbers and the `Structure` column layout stay unchanged.

## Reference and scope

The unmodified [CommonMark 0.31.2 examples](https://spec.commonmark.org/0.31.2/spec.json)
come from spec revision `9103e341a973013013bb1a80e13567007c5cef6f`.
The JSON SHA-256 is
`d431b29d97b6f73e69d547109cf5081578fac931e72afe95639ebe766c1b2a20`.
John MacFarlane's spec examples are CC-BY-SA-4.0; the upstream notice is
`analysis/structure/tests/data/LICENSE.commonmark`. The transformed fixture
keeps that licence. Hex encoding preserves every input byte, including
control characters, without embedding control characters in source files.

The offline oracle is [cmark 0.31.1](https://github.com/commonmark/cmark/tree/bb3678d7a73cb02d35c8876ecd097072636200a8),
revision `bb3678d7a73cb02d35c8876ecd097072636200a8`, BSD-2-Clause.
Its source archive SHA-256 is
`3da93db5469c30588cfeb283d9d62edfc6ded9eb0edc10a4f5bbfb7d722ea802`.
`analysis/structure/reference/commonmark.py` first requires cmark to render
all 652 published HTML outputs exactly. It then extracts cmark's XML block
trees, heading levels and tightness flags into the committed TSV. The TSV
SHA-256 is `16b52a93d83391f3071d0a30bb4184dc887c75f1bc54982731ee7707f86a5e2c`.
No reference executable or Python package is needed by the Rust tests.

The comparison removes cmark thematic-break nodes, which this layer does
not emit, and our section wrappers, which cmark does not emit. It checks
all remaining parent indexes, kinds, heading levels and list flags. It
does not claim inline rendering or exact agreement with cmark's source
positions. Separate offset tests validate UTF-8 spans and project the new
blocks and table cells through whitespace normalization, then back to the
original text. The graph codec also round-trips the new kinds.

| Corpus section | Matching block trees | Total |
| --- | ---: | ---: |
| HTML blocks | 44 | 44 |
| Block quotes | 25 | 25 |
| List items | 48 | 48 |
| Lists | 25 | 26 |
| ATX headings | 18 | 18 |
| Setext headings | 26 | 27 |
| Indented and fenced code | 41 | 41 |
| All spec examples | 577 | 652 |
| Baseline `0edf2f35`, all examples | 500 | 652 |

The 75 remaining differences are 74 examples containing link reference
definitions, which stay paragraphs, and example 96, where the deliberate
front-matter extension consumes the opening YAML-like block. They are
listed in the test assertion. A new failure, or an unexpected change in
that list, fails the test. Matching an inline-only example means only that
its surrounding block tree matches.

Footnotes and plain-text column tables are extensions, outside CommonMark.
Footnote bodies are opaque and indented 4 columns relative to their current
container; footnote references are not resolved. Plain tables need at least
2 aligned rows and 2 columns, separated by multiple spaces or a tab. This
is a conservative text heuristic, not a terminal-width or PDF layout reader.

## Verification

The structure suite includes 652 reference examples, CR/LF/CRLF and Unicode
offset checks, graph round trips, normalization projection and mixed block
combinations. Server codec tests preserve old tag numbers and round-trip
new tags. Grouping continues to classify loose lists as navigation.

To regenerate the oracle, pass the downloaded spec JSON, the pinned cmark
executable and an output file to `analysis/structure/reference/commonmark.py`.
Run the committed corpus with `cargo test --locked -p turbo-structure`.

## Release timing and remaining gate

Baseline `0edf2f35` and this branch used the same benchmark example,
release profile, AMD Ryzen 9 9950X3D host and CPU affinity (logical CPU 4).
Each process discarded 3 warmups and reported the median of 201 parses.
There were 24 pairs, alternating baseline/branch order. The table reports
the median of those process medians, in milliseconds. A separate CRF
training process ran during measurement; these are local timing records,
not a hardware-independent speed claim.

| Input and syntax | Baseline ms | Branch ms | Branch / baseline |
| --- | ---: | ---: | ---: |
| Baseline analysis document, Markdown | 0.3309655 | 0.3174495 | 0.9592 |
| Same document, plain text | 0.1989055 | 0.2573960 | 1.2941 |
| UD English EWT training sentences, plain text | 1.4777415 | 1.3932770 | 0.9428 |

The no-slowdown gate is **not fully met**. The Markdown and prose paths
improve, but reading Markdown source as plain text is 29.4% slower. That
input produces no tables, so this is candidate-detection overhead, not
extra output work. Issue #948 stays open pending that optimization.

The analysis-document input is `docs/analysis.md` at the baseline revision,
324322 bytes, SHA-256
`1b3145c00bcde19da0ba88a49cb3b5eb4ef5c84ec74533374f1318c65309b28c`.
The prose input joins the `# text = ` fields from UD English EWT r2.16
training data with 2 newlines and a final newline. Its SHA-256 is
`f68a568dd94566c9d5b8b207bdd5c9603f281fdea09239136fb3bca76ee14676`.
The EWT source is CC-BY-SA-4.0, SHA-256
`3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b`.
No corpus copy is added by this change.

Reproduce each process with the release `block_bench` example, passing
`TEXT markdown|plain 201`; alternate the two binaries on the same CPU.
All 57 structure tests, 160 related library tests, 4 tree tests and clippy
with warnings denied passed. The 75 explicit CommonMark gaps remain.
