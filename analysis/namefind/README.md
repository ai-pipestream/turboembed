# turbo-namefind

OpenNLP 3.x's name finders that need no model, ported to Rust:
`DictionaryNameFinder`, `RegexNameFinder` with the patterns of
`RegexNameFinderFactory`, and `NameFinderME.dropOverlappingSpans`, which merges
the names of several finders. The crate documentation (`src/lib.rs`) describes
the modules, what a Java pattern may use, and where OpenNLP's behaviour is
surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork, and `java.util.regex` in JDK 21.

## Golden files

`tests/golden.rs` compares the crate with goldens a JVM running OpenNLP wrote:
the `namefind` runs of `analysis/goldens-tool` (`NameFindGolden.java`), made
in its container by `turbo-goldens generate` and checked by
`turbo-goldens verify`. Nothing calls the JVM at test time. Each run reads the
input `analysis/goldens/data/inputs/namefind/sentences.txt` and writes
`analysis/goldens/data/namefind/<run>/namefind/sentences.golden`.

| File | Contents |
|---|---|
| `inputs/namefind/sentences.txt` | 31 sentences, tokens separated by single spaces: phone numbers, emails (quoted, IP literal, U+017F and U+212A in the local part), URLs (every prefix, ports, percent escapes, nonspacing marks and characters outside the BMP after the host), MGRS references, coordinates with U+00B0, U+2019 and U+201D, and the dictionaries' entries in several cases |
| `fixtures/namefind/*.xml` | `places` (case-insensitive, multi-token entries that overlap, U+0130, U+212A, an emoji), `people` (case-sensitive, two entries that differ only in case), `overlap` (no case attribute, overlapping entries of one to four tokens) and `empty` |
| `fixtures/namefind/patterns.txt` | 120 Java patterns with the flags they are compiled with and what this port does with them: `ok` (translated), `unsupported` (refused) or `syntax` (Java refuses it) |

| Run | Contents |
|---|---|
| `generated` | 3,000 generated sentences for the regex finders (`f`), built from random emails, URLs, phone numbers, MGRS references, coordinates and words, with random ASCII punctuation and non-ASCII characters inserted, and 1,000 for the dictionaries (`d`) from the entries' tokens in several cases |
| `dictionary` | Each dictionary's names for the hand-made and `d` sentences |
| `regex-builtin` | Each factory pattern alone, all five together (with the order of their types) on the hand-made and `f` sentences, and a config map over all five on the hand-made ones |
| `regex-factory` | The type order of the factory's map for every subset of the defaults, two orders with repeats, and 400 random config maps of up to 59 types, over no, empty, all or some defaults |
| `regex-user` | Whether Java compiles each pattern of `patterns.txt`, and its names on the hand-made sentences and the first 400 `f` sentences |
| `merge` | `dropOverlappingSpans` on 3,000 random span lists with types that sort differently by UTF-16 unit and by code point, and on the names of all four dictionaries and the five patterns together, for every sentence with names |

A sentence key is `h` (hand-made, by line), `b` (boundary fixtures), `f` or `d` and its index. A name is
a row with its finder or pattern, its sentence key, its token span and its
type, in the order the finder returns them; a sentence with no names has no
row. Every run draws the generated sentences and its random cases from one
seeded `Random`, so the `f` and `d` sentences are the same in every run.

The type order of the factory's map is read from `RegexNameFinder`'s private
`regexMap` field by reflection.

A user pattern whose translation refuses a sentence (`Error::Text`, for the
supplementary constructs `src/java_regex/mod.rs` lists) is not compared on that sentence;
the test counts them and fails if they are more than 2% of the comparisons
(265 supplementary-text cases out of 31,543 comparisons). BMP texts must have zero refusals.

### Java boundary and end assertions

The user-pattern matcher follows JDK 21 for `\b` / `\B`: ASCII word
characters plus nonspacing marks with a letter or digit base. The
backward walk follows UTF-16 semantics, including supplementary-mark
asymmetry. `$` and `\Z` accept the end or the position before the final
line terminator (CRLF is one terminator); `\z` requires the end. BMP
texts no longer fail a text-limit check. Patterns that can start inside
UTF-16 surrogate pairs still return `TextLimit::Supplementary`.

The `regex-user` golden also reads `fixtures/namefind/regex-boundaries.txt`:
TAB-separated tokens, with `\n` and `\r` escapes for embedded line endings.
The file includes Hindi, Thai, Arabic, NFD Latin and supplementary-mark
cases. Its digest is recorded alongside the patterns digest in the golden.
`NameFindGolden.main` regenerates this run without unrelated statistical
models. Compile the current `NameFindGolden.java` against the pinned
`turbo-goldens-java` jar, then run:

```text
java -cp <compiled-classes>:<goldens-jar> ai.pipestream.turbo.goldens.NameFindGolden \
  <fixtures-dir> <namefind-sentences.txt> <output.golden> <opennlp-commit>
```

Reference semantics: [OpenJDK 21 Pattern](https://github.com/openjdk/jdk21u/blob/master/src/java.base/share/classes/java/util/regex/Pattern.java).
The legacy regex-translation oracle only compares texts that it supports;
the newly supported boundaries are checked against the JVM golden instead.
`cargo run --locked --release -p turbo-namefind --example regex_timing -- 150`
measures the ASCII subset of the checked-in golden sentences with five
user patterns, excluding compilation and fixture loading.

On an AMD Ryzen 9 9950X3D 16-Core Processor, Linux x86_64, release builds of main
and this change were pinned to one allowed CPU and alternated for eight
rounds per arm. Each round made 3,982,500 calls over 1,593 ASCII sentences
and returned 753,000 matches. Median ns/call: main 365.24 (361.62-414.23),
updated 367.80 (363.90-394.48), a 0.70% difference within the overlapping
run ranges. This receipt does not establish a speedup. Use 500 iterations
with the example above to reproduce the workload.
