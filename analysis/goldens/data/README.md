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

## Goldens

`<annotator>/<model>/<set>/<id>.golden` is what the OpenNLP annotator
`<annotator>` produced with model `<model>` (the model file's name
without `.bin`) on `inputs/<set>/<id>.txt`. Each file's header names the
OpenNLP commit, the JDK, the input's hash and the model's hash.

Where a token or sentence starts or ends between the two halves of a
surrogate pair, its row holds the whole character's bytes and the golden
also has `turbo:utf16<opennlp:tokens>` (or `<opennlp:sentences>`) with
the JVM's exact UTF-16 spans; only
`tokenize/opennlp-en-ud-ewt-tokens-1.3-2.5.4/en/emoji-split.golden` has
one now. A golden whose annotator threw on its input has an `error`
record and no layers; none of the checked in goldens has one, as none of
the annotators throws on these inputs. `docs/analysis.md` ("Generating
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

## Decision point dumps

`<annotator>/<model>/<set>/<id>.dump`, beside each golden of
`sentdetect`, `tokenize` and `langdetect`, records every
decision the annotator asked its maxent model for on that input: the
context strings, at the UTF-8 byte offset the decision is about, and the
probability of each outcome the model returned. `docs/analysis.md`
("Decision point dumps") describes the format and how each annotator's
decisions are paired with positions. `generate` writes the dumps
together with the goldens, and the same rule holds: do not edit them by
hand.

The dumps checked in now were made by running the Java generator
directly with `--dumps` (see `analysis/goldens-tool/README.md`) under the
JDK named in their headers, in the same run as the goldens, with the
OpenNLP build and models named there. A normalizer asks no model, so the
`normalize` goldens have no dumps; neither do the `numbers` goldens. The run in the pinned container,
through `turbo-goldens generate` and `verify`, is still to be done on a
machine with Docker.
