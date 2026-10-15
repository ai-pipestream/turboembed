# Sentence segmentation conformance suites

Test data for the sentence detectors, read by `turbo_goldens::sentences`
and run by `analysis/sentdetect/tests/conformance.rs`. Each directory
keeps the licence of its source.

| Directory | Source | Licence |
|---|---|---|
| `pragmatic-segmenter/` | the specs of Pragmatic Segmenter (`github.com/diasks2/pragmatic_segmenter`, commit 358fe97e62902cd86fa6b85063c3bf91aa6c156a, `spec/pragmatic_segmenter/languages/*_spec.rb`); `english_spec.cases` 001 to 052 are the Golden Rules | MIT, Copyright (c) 2015 Kevin S. Dias |
| `pysbd/` | the tests of pySBD (`github.com/nipunsadvilkar/pySBD`, commit 5905f13be4fc95f407b98392e0ec303617a33d86, `tests/`) | MIT, Copyright (c) 2019 Nipun Sadvilkar |
| `uax29/` | `SentenceBreakTest.txt` of the Unicode Character Database 18.0.0 (`ucd/auxiliary/`), copied unchanged | Unicode License v3 |

## The `.cases` format

The specs and tests were turned into plain text case files, one per
source file, so that no Ruby or Python is needed to read them. Every
expectation of the source (an input text and the sentences the
segmenter must return) is one case:

```text
case pragmatic-segmenter/english_spec/001
lang en
name Golden Rules: Simple period to end sentence #001
option doc_type=pdf
text Hello World. My name is Jonas.
sentence Hello World.
sentence My name is Jonas.
end
```

- `case`: `<suite>/<source file>/<number>`, numbered in file order.
- `lang`: the language the source's test gives the segmenter.
- `name`: the test's name, or the name of the list it comes from.
- `option`: repeatable. The segmenter options the test passes
  (`clean=true`, `clean=false`, `doc_type=pdf`, `doc_type=docx`), and
  `skipped-upstream` or `xfail-upstream` for a test the source skips or
  expects to fail.
- `text`, `sentence`: written with the escapes of `turbo_goldens::escape`
  (`\n`, `\r`, `\t`, `\\`, and `\uXXXX` for the other control
  characters), so every value fits on one line.

The texts and sentences are the source's own, unchanged apart from the
escaping. Some cases expect sentences that the source's cleaning step
rewrote (with `clean=true` or a `doc_type`): the expected sentences are
not the text cut into pieces. No detector here cleans text, so such a
case fails for every detector, and its expected failure line says that
the expected sentences rewrite the text.

The conversion was done once with a script that is not part of this
repository; the files are the data and are edited by hand if a source
moves.
