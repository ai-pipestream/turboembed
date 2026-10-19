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
| `fixtures/namefind/patterns.txt` | 114 Java patterns with the flags they are compiled with and what this port does with them: `ok` (translated), `unsupported` (refused) or `syntax` (Java refuses it) |

| Run | Contents |
|---|---|
| `generated` | 3,000 generated sentences for the regex finders (`f`), built from random emails, URLs, phone numbers, MGRS references, coordinates and words, with random ASCII punctuation and non-ASCII characters inserted, and 1,000 for the dictionaries (`d`) from the entries' tokens in several cases |
| `dictionary` | Each dictionary's names for the hand-made and `d` sentences |
| `regex-builtin` | Each factory pattern alone, all five together (with the order of their types) on the hand-made and `f` sentences, and a config map over all five on the hand-made ones |
| `regex-factory` | The type order of the factory's map for every subset of the defaults, two orders with repeats, and 400 random config maps of up to 59 types, over no, empty, all or some defaults |
| `regex-user` | Whether Java compiles each pattern of `patterns.txt`, and its names on the hand-made sentences and the first 400 `f` sentences |
| `merge` | `dropOverlappingSpans` on 3,000 random span lists with types that sort differently by UTF-16 unit and by code point, and on the names of all four dictionaries and the five patterns together, for every sentence with names |

A sentence key is `h` (hand-made, by line), `f` or `d` and its index. A name is
a row with its finder or pattern, its sentence key, its token span and its
type, in the order the finder returns them; a sentence with no names has no
row. Every run draws the generated sentences and its random cases from one
seeded `Random`, so the `f` and `d` sentences are the same in every run.

The type order of the factory's map is read from `RegexNameFinder`'s private
`regexMap` field by reflection.

A user pattern whose translation refuses a sentence (`Error::Text`, for the
three constructs `src/java_regex.rs` lists) is not compared on that sentence;
the test counts them and fails if they are more than 2% of the comparisons
(about 1.2% now).
