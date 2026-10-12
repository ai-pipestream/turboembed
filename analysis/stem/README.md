# turbo-stem

OpenNLP 3.x's stemmers, ported to Rust: `PorterStemmer`, the 21 Snowball
stemmers of `opennlp.tools.stemmer.snowball`, the 16 light and minimal
stemmers of `opennlp.tools.stemmer.light` (the two Norwegian ones for each of
three choices of written standard), `CachingStemmer`, and `StemmerAnnotator`
(here `StemAnnotator`) as an analysis step that reads `opennlp:tokens` and
writes `turbo:stems` (the layer is in `turbo-stem-layers`). The crate documentation
(`src/lib.rs`) describes the modules and where OpenNLP's behaviour is
surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork. The light and minimal stemmers there are the same files as on
the `ready/light-stemmers` branch of opennlp-addons (commit
`14bcaffef2c1c4a579a2a8df03edbd09a1193192`).

The Snowball stemmers are generated code; `snowball/README.md` says from what
and how.

## Golden files

`tests/golden.rs` compares every stemmer with files a JVM running OpenNLP
wrote. The Java program, its inputs and its outputs are under `golden/`;
nothing calls the JVM at test time. `golden/generate.sh` rebuilds the outputs
from OpenNLP built at the commit above and installed into the local Maven
repository, with JDK 21; two runs write the same bytes. Given a snowball-data
checkout at the commit below, it first takes the samples again.

| Path | Made by | Contents |
|---|---|---|
| `inputs/<language>.txt` | `generate.sh` | About 1,000 words of the language's Snowball reference vocabulary (`voc.txt` of snowball-data at `da07690d20ce69ee16b5fb95796f92090cc9b4a1`): every n-th line from the first, for the 16 languages whose data is under snowball-data's BSD license. Arabic (GPL 3), Catalan, French, Greek and Indonesian (in part from Wikipedia, CC BY-SA) have none |
| `snowball/<language>.txt` | `generate.sh` | The same lines of `output.txt`: Snowball's own stems of those words |
| `inputs/edge.txt` | by hand | 135 words every stemmer gets: the empty word, short words and bare suffixes, upper and mixed case, digits, apostrophes and hyphens, combining accents, words of several scripts, a word of 104 letters, supplementary characters, and lone surrogates |
| `inputs/generated/<language>.txt` | `StemGolden.java` | 1,000 words per language (and for `porter`): a random base of the language's letters with one or two endings from its Snowball suffix tables (read from the generated Java class) and its light stemmers' endings, some with an upper-case letter, a digit, an apostrophe, a hyphen, a combining accent, a supplementary character or a lone surrogate mixed in |
| `expected/<stemmer>.txt` | `StemGolden.java` | The stem OpenNLP returns for each word of its inputs, one per line, after a header naming the Java stemmer, the OpenNLP commit and the input files in the order they are read |

There are 46 stemmers: `porter` (`PorterStemmer`), `snowball-<language>` for
each `SnowballStemmer.ALGORITHM`, `snowball-<language>-repeat2` for English,
German, Russian and Turkish with a repeat count of 2, and one per light or
minimal stemmer (`norwegian-light-bokmaal`, `-nynorsk` and `-both`, and the
same for the minimal one). Each gets the sample of its language when there is
one, `edge.txt`, and the generated words of its language, through one Java
instance in that order: 90,713 words in all.

A word or stem is one line: the backslash, tab, line feed, carriage return,
other C0 controls, lone surrogates and a leading `#` are escaped (`\\`, `\t`,
`\n`, `\r`, `\uXXXX`, one UTF-16 unit each); everything else is UTF-8. A line
of an input file that starts with `#` is a comment.

`tests/golden.rs` also checks the Snowball stemmers against the samples in
`snowball/`.

## The full Snowball vocabularies

`tests/snowball_vocabulary.rs` stems every word of snowball-data (10.4 million
words, most of them Arabic) when `TURBO_SNOWBALL_DATA` names a checkout of
it, and passes without checking anything otherwise:

```sh
TURBO_SNOWBALL_DATA=/path/to/snowball-data cargo test --release -p turbo-stem --test snowball_vocabulary -- --nocapture
```

At snowball-data commit `da07690d20ce69ee16b5fb95796f92090cc9b4a1` (January
2025, which updated the data for the Snowball commit OpenNLP's classes were
generated from) every word of all 21 vocabularies stems to its line of
`output.txt`.

Snowball has changed several algorithms since. At snowball-data commit
`a0ec0d0a2839ec885878868de20fcb63209d92b0` (June 2026), these words of the
vocabulary of that commit stem differently, so OpenNLP's stems are not current
Snowball's for them:

| Language | Words that differ | Of | Examples (OpenNLP, then current Snowball) |
|---|---:|---:|---|
| Danish | 35 | 23,868 | `bbc's` bbc's / bbc |
| Dutch | 20,897 | 45,670 | `aalmoezen` aalmoez / aalmoes: Snowball's `dutch` is now the Kraaij-Pohlmann algorithm, and the one OpenNLP has is `dutch_porter` |
| English | 50 | 42,649 | `emergency` emerg / emergenc |
| Finnish | 183 | 84,399 | `abachan` abachan / abach |
| French | 838 | 21,653 | `bijoux` bijoux / bijou, `c'est` c'est / est |
| German | 915 | 35,053 | `abgeleitet` abgeleitet / abgeleit |
| Italian | 1,867 | 37,426 | `all'acqua` all'acqu / acqua |
| Norwegian | 154 | 20,913 | `binders` bind / binders |
| Spanish | 9 | 28,378 | `educacion` educacion / educ |
| Swedish | 506 | 30,738 | `aftonbladet` aftonbladet / aftonblad |
| Turkish | 78 | 96,325 | `a'ya` a / a'ya |

Arabic, Catalan, Greek, Hungarian, Indonesian, Irish, Portuguese, Romanian,
Russian and Snowball's Porter agree with current Snowball on every word of
their current vocabularies.
