# turbo-stem

OpenNLP 3.x's stemmers and layered term analyzer, ported to Rust:
`PorterStemmer`, the 21 Snowball stemmers of `opennlp.tools.stemmer.snowball`,
the 16 light and minimal stemmers of `opennlp.tools.stemmer.light` (the two
Norwegian ones for each of three choices of written standard),
`CachingStemmer`, `StemmerAnnotator` (here `StemAnnotator`) as an analysis step
that reads `opennlp:tokens` and writes `turbo:stems`, and the layered term
analyzer (`Dimension`, `Term`, `TermAnalyzer`, `NormalizationProfile`,
`NormalizationProfiles`, and `TermAnnotator`, which reads `opennlp:tokens` and
optional `opennlp:language`, `opennlp:pos`, and `opennlp:lemmas` and writes
`turbo:terms`; both `Stems` and `Terms` are defined in `turbo-analysis-layers`
and re-exported by `turbo-stem-layers`). The crate documentation (`src/lib.rs`)
describes the modules and where OpenNLP's behaviour is surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork. The light and minimal stemmers there are the same files as on
the `ready/light-stemmers` branch of opennlp-addons (commit
`14bcaffef2c1c4a579a2a8df03edbd09a1193192`).

The Snowball stemmers are generated code; `snowball/README.md` says from what
and how.

## Snapshots

`tests/snapshots.rs` freezes every stemmer's output under `TURBO_BLESS=1`:
one TSV per (stemmer, input) under `tests/snapshots/`, each line a word and
its stem. A missing snapshot fails. Inputs live under `tests/inputs/` (the
Snowball sample words, `edge.txt`, and generated words). `tests/properties.rs`
holds pure-Rust checks (UTF-8/borrow equivalence, Porter length and
idempotence, Snowball no-panic, light stemmers never grow short ASCII, and
`Stems::check` with per-token stem equality).

`tests/snowball_vocabulary.rs` and the samples under `tests/snowball/` still
compare Snowball stemmers with Snowball's own `output.txt` samples.

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
