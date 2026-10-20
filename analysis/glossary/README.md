# turbo-glossary

OpenNLP's glossary term matching (`opennlp.tools.glossary`) and glossary
readers (`opennlp.tools.formats.glossary`), ported to Rust: a character
matcher, a token matcher, their composite, the analysis step that writes
`turbo:glossary-terms`, and readers of comma separated, tab separated and TBX
glossaries, with a TBX writer. The crate documentation (`src/lib.rs`) describes
the matchers, where they differ from OpenNLP and the OpenNLP behaviour that is
surprising. The glossary and the layer are in `turbo-glossary-layers`, so a
consumer of the matches does not depend on this crate.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork, run on JDK 21.

## Golden files

`tests/jvm_goldens.rs` compares the crate with goldens a JVM running OpenNLP
wrote: the `glossary` runs of `analysis/goldens-tool` (`GlossaryGolden.java`),
made in its container by `turbo-goldens generate` and checked by
`turbo-goldens verify`. Nothing calls the JVM at test time. A run's golden is
`analysis/goldens/data/glossary/<run>/glossary/<input>.golden`; the inputs are
the files of `analysis/goldens/data/inputs/glossary/`, all written for this
project, and the matchers' glossary is the fixture
`analysis/goldens/data/fixtures/glossary-terms.tsv` (40 entries, read with
`CsvGlossaryReader('\t', true)`).

| Run | Inputs | Contents |
|---|---|---|
| `exact`, `ignore-case` | `text-*` | The matches of `AhoCorasickGlossaryMatcher` with `ignoreCase` false and true: byte span and entry (the fixture row of the match's id and term) |
| `whitespace+ignore-case` | `text-*` | The same with `WhitespaceCharSequenceNormalizer` folding terms and text |
| `analyzed` | `text-*` | The matches of `TermAnalyzingGlossaryMatcher` with a `TermAnalyzer` that lowercases and stems with `PorterStemmer`, over the entries whose terms give at least one token and no blank one; the analyzer's tokens of the text with their values (`turbo:analyzed-terms`) and of each entry (`turbo:analyzed-entries`) |
| `composite` | `text-*` | `CompositeGlossaryMatcher` of the exact matcher over every entry, then the analyzed one, with the same analysis layers |
| `csv` | `csv-*` | The entries `CsvGlossaryReader()` reads, or its exception |
| `tsv-header` | `tsv-*` | The same for `CsvGlossaryReader('\t', true)` |
| `tbx-en`, `tbx-de` | `tbx-*` | The same for `TbxGlossaryReader("en")` and `("de")` |

The texts hold terms with other case, other spacing, inflections, hyphens and
punctuation around them, terms inside longer words, nested and overlapping
terms, accented, Greek, Turkish, CJK and Korean text, emoji and characters
outside the BMP. The delimited inputs cover quoting, doubled quotes, unclosed
quotes, text after a closing quote, a quote inside an unquoted field, missing
columns, blank ids and terms, a line of spaces, a header only and an empty
file. The TBX inputs cover TBX 2 and 3, a prefixed namespace, the wrong
namespace, a root that is not TBX, entities and character references, an
external entity, an undeclared entity, an unbound prefix, two roots, terms
outside the body, entries without an id or with a namespaced one, blank
terms, and malformed XML.

The test checks each matcher's matches, and the analysis it matches by,
against the goldens; for the readers, the entries, or that both sides refuse
the input and, where the JVM's message names a line, that the line is the
same. The whitespace run scans the folded text and maps each hit back through
the view's alignment before checking the boundaries in the source, as
OpenNLP does inside its matcher.

`tests/round_trip.rs` writes every glossary the readers take from these
inputs, and the fixture, as TBX in both dialects and reads it back to the same
entries. `tests/pipeline.rs` runs the step over a normalized view, and
`tests/random.rs` reads mutated copies of the inputs and checks the character
automaton against a naive search on random glossaries and texts.

## Timing

`examples/glossary_bench.rs` finds terms at token edges both ways, over the
same made-up glossary and text of about 4 MB (UAX #29 word tokens): the
character automaton with the tokens' edges as boundaries, and the phrase
matcher of turbo-matcher over the tokens' text. Both find the same matches.
On a 4 core Intel Xeon at 2.1 GHz, release build, best of five:

| Terms | Matcher | Build | Match | Heap |
|---|---|---|---|---|
| 1,000 | character automaton | 1.7 ms | 109 ms (37 MB/s) | 0.6 MB |
| 1,000 | phrase matcher | 0.5 ms | 32 ms (127 MB/s) | 0.2 MB |
| 50,000 | character automaton | 97 ms | 269 ms (15 MB/s) | 20.5 MB |
| 50,000 | phrase matcher | 18 ms | 55 ms (73 MB/s) | 3.9 MB |

The phrase matcher looks each token up once and walks a trie of token ids; the
automaton takes a hash table step per character. So matching by tokens uses the
phrase matcher, and the character automaton is kept for what only it does:
matching the separators inside a term exactly, as OpenNLP's
`AhoCorasickGlossaryMatcher` does.

## Automaton images

`MappedGlossaryWriter` writes the character automaton as one flat image, a
double-array Aho-Corasick automaton (the cell layout and the checks are in
`src/mapped/mod.rs`), and `MappedGlossaryMatcher::from_bytes` reads it in
place: from an `ImageBuf`, from any 4-byte aligned slice, or from a file mapped
with `MappedFile` (feature `mmap`). The image is the in-heap automaton's states
laid out by code, so `scan` and `find` give the same hits in the same order
and the same matches; `tests/mapped.rs` checks that on random glossaries and
on the goldens' glossary and inputs, in both cases and with a normalizer.
Loading checks the whole image (counts, checksum, every code and cell link,
depths that make every walk end, the payload's UTF-8) and allocates nothing;
`src/mapped/tests.rs` refuses every truncation and every flipped bit, and
changes each word of an image to eight other values, with the checksum
resealed: each change is refused, or still matches without a panic.
`tests/mapped_alloc.rs` counts allocations: none for loading, and none for
matching once the caller's `Scratch` and output have grown.

`examples/mapped_bench.rs` times both on the same glossary and about 8 MB of
text, the two matchers alternating run by run (median of 7 runs, five
processes each). Zen 5 desktop, one pinned core, release build, load average
9 to 22 of 32 threads from other work:

| Terms | Case | Heap | Image | Load | Scan, heap | Scan, image | Find, heap | Find, image |
|---|---|---|---|---|---|---|---|---|
| 50,000 | sensitive | 20.5 MB | 11.4 MB | 2.3 ms | 305 ms | 182 ms | 361 ms | 204 ms |
| 50,000 | insensitive | 20.5 MB | 11.3 MB | 2.3 ms | 551 ms | 342 ms | 576 ms | 289 ms |
| 1,000,000 | sensitive | 328 MB | 181 MB | 60 ms | 1061 ms | 552 ms | 1094 ms | 630 ms |
| 1,000,000 | insensitive | 328 MB | 179 MB | 65 ms | 1162 ms | 641 ms | 1176 ms | 651 ms |

Every cell of the double array holds a state (6,689,398 for the million
terms): breadth-first placement, codes numbered by use, leaves no gaps here.
Building the in-heap automaton for a million terms takes about 2.4 s and
writing its image about 1.3 s; loading the image takes 60 ms, most of it the
checksum and the cell checks.

