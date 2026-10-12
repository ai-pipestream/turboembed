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
  Java), lines holding only whitespace, and paragraphs.
- `inputs/multi/`: one text each in German, French, Spanish, Portuguese,
  Italian, Dutch, Russian, Japanese, Chinese and Arabic, and a mixed
  script text with emoji, CJK Extension B characters and mathematical
  letters outside the Basic Multilingual Plane.

## Goldens

`<annotator>/<model>/<set>/<id>.golden` is what the OpenNLP annotator
`<annotator>` produced with model `<model>` (the model file's name
without `.bin`) on `inputs/<set>/<id>.txt`. Each file's header names the
OpenNLP commit, the JDK, the input's hash and the model's hash.

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

## Decision point dumps

`<annotator>/<model>/<set>/<id>.dump`, beside each golden, records every
decision the annotator asked its maxent model for on that input: the
context strings, at the UTF-8 byte offset the decision is about, and the
probability of each outcome the model returned. `docs/analysis.md`
("Decision point dumps") describes the format and how each annotator's
decisions are paired with positions. `generate` writes the dumps
together with the goldens, and the same rule holds: do not edit them by
hand.

The dumps checked in now were made by running the Java generator
directly with `--dumps` (see `analysis/goldens-tool/README.md`) under the
JDK named in their headers, with the OpenNLP build and models the goldens
were made with; the goldens of that run are byte for byte the checked in
ones. The run in the pinned container, through `turbo-goldens generate`
and `verify`, is still to be done on a machine with Docker.
