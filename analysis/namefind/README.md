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

`tests/golden.rs` compares the crate with files a JVM running OpenNLP wrote. The
Java program, its inputs and its outputs are under `golden/`; nothing calls the
JVM at test time. `golden/generate.sh` rebuilds the outputs from OpenNLP built
at the commit above and installed into the local Maven repository; two runs
write the same bytes. This generator is standalone for now, like the one in
`analysis/sequence/golden/`.

| Path | Made by | Contents |
|---|---|---|
| `inputs/sentences.txt` | by hand | 31 sentences, tokens separated by single spaces: phone numbers, emails (quoted, IP literal, U+017F and U+212A in the local part), URLs (every prefix, ports, percent escapes, nonspacing marks and characters outside the BMP after the host), MGRS references, coordinates with U+00B0, U+2019 and U+201D, and the dictionaries' entries in several cases |
| `inputs/dictionaries/*.xml` | by hand | `places` (case-insensitive, multi-token entries that overlap, U+0130, U+212A, an emoji), `people` (case-sensitive, two entries that differ only in case), `overlap` (no case attribute, overlapping entries of one to four tokens) and `empty` |
| `inputs/patterns.txt` | by hand | 114 Java patterns with the flags they are compiled with and what this port does with them: `ok` (translated), `unsupported` (refused) or `syntax` (Java refuses it) |
| `expected/sentences.txt` | `NameFindGolden.java` | 3,000 generated sentences for the regex finders (`f`), built from random emails, URLs, phone numbers, MGRS references, coordinates and words, with random ASCII punctuation and non-ASCII characters inserted, and 1,000 for the dictionaries (`d`) from the entries' tokens in several cases |
| `expected/dictionary.txt` | `NameFindGolden.java` | Each dictionary's names for the hand-made and `d` sentences |
| `expected/regex-builtin.txt` | `NameFindGolden.java` | Each factory pattern alone, all five together (with the order of their types) on the hand-made and `f` sentences, and a config map over all five on the hand-made ones |
| `expected/regex-factory.txt` | `NameFindGolden.java` | The type order of the factory's map for every subset of the defaults, two orders with repeats, and 400 random config maps of up to 59 types, over no, empty, all or some defaults |
| `expected/regex-user.txt` | `NameFindGolden.java` | Whether Java compiles each pattern of `inputs/patterns.txt`, and its names on the hand-made sentences and the first 400 `f` sentences |
| `expected/merge.txt` | `NameFindGolden.java` | `dropOverlappingSpans` on 3,000 random span lists with types that sort differently by UTF-16 unit and by code point, and on the names of all four dictionaries and the five patterns together, for every sentence with names |

A sentence key is `h` (hand-made, by line), `f` or `d` and its index. A line of
names is the sentence key and each name as `start end type`, separated by tabs,
in the order the finder returns them; a sentence with no names has no line.
Strings are escaped: every UTF-16 code unit outside U+0020..U+007E, and the
backslash, is `\uXXXX`, and tab, line feed and carriage return are `\t`, `\n`
and `\r`.

The type order of the factory's map is read from `RegexNameFinder`'s private
`regexMap` field by reflection.

A user pattern whose translation refuses a sentence (`Error::Text`, for the
three constructs `src/java_regex.rs` lists) is not compared on that sentence;
the test counts them and fails if they are more than 2% of the comparisons
(about 1.2% now).
