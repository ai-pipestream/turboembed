# Golden data

This directory holds the input texts the golden files are made from and
the golden files themselves. `docs/analysis.md` describes the golden
format.

## Inputs

`inputs/<set>/<id>.txt` are UTF-8 texts. A file's bytes are the text:
there is no byte order mark and the final line end is part of the text.

The texts were written for this project to exercise the annotators, and
are not taken from any corpus.

- `inputs/en/`: English texts with abbreviations ("Dr.", "U.S."), straight
  and curly quotes, numbers with periods, ellipses, URLs and an email
  address, a numbered and a bulleted list, all caps sentences, emoji
  (characters outside the Basic Multilingual Plane, so surrogate pairs in
  Java), lines holding only whitespace, and paragraphs. `emoji-split.txt`
  puts an emoji after a quote, a comma and an ampersand, where
  `TokenizerME` splits it between its two UTF-16 units.
- `inputs/multi/`: one text each in German, French, Spanish, Portuguese,
  Italian, Dutch, Russian, Japanese, Chinese and Arabic, and a mixed
  script text with emoji, CJK Extension B characters and mathematical
  letters outside the Basic Multilingual Plane.
- `inputs/edge/`: 34 short texts for the edge cases of the segmenters:
  abbreviations, numbers, quotes and brackets, dashes and hyphens,
  ellipses, URLs and e-mail addresses, apostrophes, CR LF and lone CR,
  tabs, no-break and ideographic spaces, `U+0085`, `U+2028`, `U+2029`,
  the separators `U+001C..U+001F`, leading and trailing whitespace, an
  empty input and a whitespace only input, Spanish, French and German,
  Greek, Cyrillic, Hebrew, Arabic, Japanese, Chinese, Korean, Thai, emoji
  (single, with skin tone, ZWJ sequence, flags, keycap), combining marks,
  characters outside the Basic Multilingual Plane, and two long words (300
  letters, and 254 letters followed by an emoji so that a cut after 255
  UTF-16 units falls inside a pair).
- `inputs/surrogate/`: texts on which a tokenizer model splits between the
  two halves of a surrogate pair. The name starts with the language whose
  model splits it; every model runs every file.
- `inputs/abbreviations/`: texts for the abbreviation dictionary of a
  tokenizer: known abbreviations, units, and a run of dots that the
  dictionary check jumps over.
- `inputs/alphanumeric/mixed.txt`: letters, digits and accented letters
  for the alphanumeric optimization.
- `inputs/detokenizer/`: one token per line, each line ending with a line
  feed (an empty file is an empty token list). The part of the name before
  the first `-` names the dictionary, `fixtures/<name>-detokenizer.xml`.
- `inputs/sentence-fixture/`: 17 texts for the two fixture sentence
  models: abbreviations at the start, in upper case, after a bracket or an
  apostrophe; numbers with dots; quotes after the period; `z.B.` style
  runs and an upper case letter after a period; a split whose token runs
  on past it (`hard.Bekannt.`); a lone `.`; a text without a final
  period; whitespace only; CR LF and lone CR; leading and trailing
  whitespace; `U+0085`, `U+2028`, `U+2029`; `U+001C` and `U+001F`; emoji
  inside a sentence; a character outside the Basic Multilingual Plane
  right after a period, also at the very end.

`analysis/goldens-tool/README.md` ("Runs") lists which run reads which set.

## Fixtures

`fixtures/` holds the models and dictionaries that are not published
models. They are checked in, and `analysis/goldens-tool/src/recipe.rs`
lists each with its SHA-512; `generate` and `verify` refuse a directory
that differs.

| File | What it is |
|---|---|
| `en-abbrev.bin`, `en-plain.bin` | English sentence models trained by `analysis/sentdetect/tests/fixtures/src/TrainSentenceModel.java` (see the README there): `en-abbrev` has an abbreviation dictionary of 13 entries and `useTokenEnd=true`, `en-plain` has no dictionary and `useTokenEnd=false` |
| `tokenizer-abbreviations.bin` | a copy of `analysis/maxent/tests/fixtures/models/tokenizer.bin`, a tokenizer model OpenNLP trained on six sentences with the case-insensitive dictionary `Dr.`, `U.S.`, `e.g.` (`analysis/maxent/tests/fixtures/README.md`) |
| `english-dictionary.xml` | an abbreviation dictionary written for this project, added to the English tokenizer model |
| `latin-detokenizer.xml` | Apache OpenNLP's test resource of `DictionaryDetokenizerTest`, unchanged (license header kept) |
| `en-detokenizer.xml`, `fr-detokenizer.xml`, `pt-detokenizer.xml` | Apache OpenNLP's English, French and Portuguese detokenizer dictionaries, `opennlp-tools/lang/{en,fr,pt}/tokenizer/`, unchanged (license header kept) |
| `quotes-detokenizer.xml` | two `RIGHT_LEFT_MATCHING` entries, written for this project |

## Goldens

`<annotator>/<model>/<set>/<id>.golden` is what the OpenNLP annotator
`<annotator>` produced with model `<model>` (the model file's name
without `.bin` or `.xml`) on `inputs/<set>/<id>.txt`. A run with options
has `+` and their names after the model (`+legacy`, `+keep-new-lines`,
`+alphanumeric`, `+english-dictionary`), and a run without a model names
itself there (`simple`, `whitespace`, `uax29`, `newline`). Each file's
header names the OpenNLP commit, the JDK, the input's hash, each model's
hash and the run's options.

Where a token or sentence starts or ends between the two halves of a
surrogate pair, its row holds the whole character's bytes and the golden
also has `turbo:utf16<opennlp:tokens>` (or `<opennlp:sentences>`) with
the JVM's exact UTF-16 spans. `en/emoji-split`, eight of the
`surrogate` goldens, the `simple` goldens of inputs with characters
outside the Basic Multilingual Plane (`SimpleTokenizer` classifies UTF-16
units) and four of the `tokenizer-abbreviations` goldens have one. A golden whose annotator threw on its input has an `error`
record and no layers; `SimpleTokenizer` with `keepNewLines` throws on a
whitespace only input, and those goldens have one. `docs/analysis.md` ("Generating
golden files") describes both.

Do not edit the goldens by hand. They are regenerated with the driver in
`analysis/goldens-tool` (see its README):

```
cargo run -p turbo-goldens-tool -- generate
cargo run -p turbo-goldens-tool -- verify
```

`generate` builds nothing itself: it needs the image from
`turbo-goldens image` and runs the generator in that container with the
inputs and models mounted read only, then replaces the annotator
directories here with the new output. Adding or changing an input means
running `generate` and checking in the inputs and the goldens together.

## Normalizer goldens

`inputs/norm/` holds 61 short texts written for the normalizers: curly
quotes and dashes, line ends and Unicode spaces, hyphenated line breaks,
URLs and e-mail addresses, tweets, numbers, ligatures and normalization
forms, Hangul, German, Greek and Turkish letters, confusables, emoji,
English contractions, lone symbols and a few texts for the differences
recorded in `docs/normalizers.md`. Every character is in the Unicode 15.0
repertoire, which JDK 21 implements. `normalize/<name>/norm/<id>.golden`
holds a `turbo:view` layer with the edits OpenNLP's normalizer `<name>`
made to `inputs/norm/<id>.txt`; see `analysis/goldens-tool/README.md`.

## Number goldens

`inputs/num/` holds 60 short texts written for number scanning: US and EU
grouping and decimals, values that read differently in the two notations
(`1.234`), invalid grouping, repeated separators, `k`/`m`/`bn` suffixes and
`thousand` to `trillion` words with different spaces before them,
exponents, negatives with and without a sign boundary, thin space,
no-break space, apostrophe and Arabic separators between digits, numbers
next to letters, underscores and combining marks, non-ASCII digits (which
are not scanned), 40 digit numbers, zero with a fraction, texts that end
in a separator, and prose in several languages. Every character is in the
Unicode 15.0 repertoire. `numbers/<notation>/num/<id>.golden` holds the
`turbo:numbers` layer for `inputs/num/<id>.txt` in `latin-us` or
`latin-eu`; see `analysis/goldens-tool/README.md`. Only the `numbers`
annotator runs on this set, and it has no dumps.

## Money goldens

`inputs/money/` holds 64 short texts written for money extraction:
currency symbols before and after the amount, ISO codes and English
currency words, `k`/`M`/`bn` suffixes and scale words, negative amounts
and the places a minus sign does not count, US and European notation and
amounts that read differently in each, invalid and Indian grouping,
several amounts per text, near misses (percent, bare numbers, `HK$50`,
letters running into the amount, dates and identifiers) and non-ASCII text
around the amounts (German, French, Russian, Japanese and Chinese, Arabic,
Hindi and Thai, emoji, combining marks, non-ASCII digits).
`conversion-*.txt` hold mixes of currencies for the conversion goldens.

`rates/ecb-snapshot.csv` is the rate table the conversions use: a small
hand made file in the format of the ECB's `eurofxref-hist.csv` (newest
row first, one column per currency, `N/A` for no quote) with columns USD,
JPY, GBP, CHF, CNY, INR and RUB and 12 weekdays from 2026-03-02 to
2026-03-17. The values are invented. The ECB publishes the real series
at <https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html>;
it may be reused with the source acknowledged.

`money/<notation>/money/<id>.golden` and
`money-convert/<notation>/money/<id>.golden` both hold the
layer `turbo:money`; see
`analysis/goldens-tool/README.md`. Only the two money annotators run on
the set, and they have no dumps.

## Quantity goldens

`inputs/quantity/` holds 61 short texts written for quantity extraction:
the default units (length, mass, volume, time, speed, frequency, power,
data) attached to the number or after one space, units that are case
sensitive or read only by their first token, percentages and the word
`percent`, negative values and the places a minus does not count, US and
European notation and grouping, scale markers and exponents that are not
read, amounts that money extraction takes first, and non-ASCII text around
the values. `quantity/<notation>/quantity/<id>.golden` holds the layer
`turbo:quantities`; see `analysis/goldens-tool/README.md`. Only the
quantity annotator runs on the set, and it has no dumps.

## Decision point dumps

`<annotator>/<model>/<set>/<id>.dump`, beside each golden of
`sentdetect`, `tokenize` and `langdetect` that names a model and has no
`error` record, records every
decision the annotator asked its maxent model for on that input: the
context strings, at the UTF-8 byte offset the decision is about, and the
probability of each outcome the model returned. `docs/analysis.md`
("Decision point dumps") describes the format and how each annotator's
decisions are paired with positions. `generate` writes the dumps
together with the goldens, and the same rule holds: do not edit them by
hand.

The goldens and dumps checked in now were made in the generator container by
`turbo-goldens generate`, in one run, and `turbo-goldens verify`
reproduces them byte for byte. A normalizer asks no model, so the
`normalize` goldens have no dumps; neither do the `numbers` goldens.
