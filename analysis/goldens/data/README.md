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
- `inputs/subwords/`: 14 texts for the WordPiece encoder's BERT
  normalization: precomposed accents, the same accents as combining
  marks, marks out of canonical order, capital sigma at the end of a
  word, the dotted capital I, Hangul, kana and CJK, CJK Extension E
  ideographs on both sides of U+2B920, format characters, punctuation,
  spacing marks and a virama, emoji and symbols with no piece, special
  token strings, fullwidth letters and ligatures, and words of 100 and
  101 letters.
- `inputs/zh/`: 12 Chinese texts for the unigram segmenter: sentences
  whose segmentation is ambiguous, a news paragraph, Chinese mixed with
  Latin words, digits, a URL path and an e-mail address, Chinese and
  Western punctuation, traditional characters, characters outside the
  Basic Multilingual Plane and emoji, whitespace of every kind (CR LF,
  tab, ideographic, no-break, `U+0085`, `U+001C`, `U+2028`, `U+2029`), a
  long run without punctuation, classical text, Chinese, fullwidth and
  Roman numerals, the fixture lexicon's words (`fixture-words.txt`) and
  the words of the whitespace mode lexicon (`modes.txt`). Two files are
  not new text: `classical.txt` quotes Li Bai's poem 静夜思 and the
  opening of the Analects, both in the public domain, and the sentences
  of `ambiguity.txt` are the stock examples of segmentation ambiguity
  that Chinese word segmentation papers and textbooks repeat.
- `inputs/ja/`: 14 Japanese texts for the lattice tokenizer, written for
  this project: plain sentences, segmentation ambiguity (`東京都`,
  `今日本日`), kanji the lexicons do not list, katakana with the long
  vowel mark, hiragana runs, halfwidth katakana, Japanese mixed with
  Latin, fullwidth, Greek and Cyrillic letters, kanji and Arabic
  numerals, Japanese punctuation and brackets, characters outside the
  Basic Multilingual Plane and emoji, whitespace of every kind (tab,
  ideographic, no-break, `U+0085`, `U+001C`, CR, `U+2028`, `U+2029`), a
  long run without spaces, the fixture dictionary's tied words
  (`ties.txt`) and its surface with a comma (`quoted-surface.txt`).
- `inputs/ko/`: 10 Korean texts for the lattice tokenizer, written for
  this project: sentences with and without spaces between words, words
  the lexicons do not list, Hanja, Korean mixed with Latin words and
  numbers, compatibility and conjoining jamo, punctuation, whitespace of
  every kind, characters outside the Basic Multilingual Plane and a long
  run without spaces.
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

- `inputs/bpe/`: six texts written for OpenNLP's BPE trainer: prose
  (`prose`), words whose pairs tie (`ties`), runs of one letter whose
  pairs overlap (`overlap`), several scripts (`multilingual`), characters
  outside the Basic Multilingual Plane (`emoji`) and words holding `</w>`
  as text (`end-marker`).

- `inputs/train/sentences.txt`: the 230 sentences, one per line, the
  fixture sentence model `en-plain.bin` was trained on
  (`analysis/sentdetect/tests/fixtures/sentences.txt`), for the training
  goldens.

- `inputs/formats/`: 41 small files in OpenNLP's native sample formats,
  in CoNLL-U, CoNLL-2002, CoNLL-2003, CoNLL-X and in MASC's layout, written for this project except as
  noted below; the part of the id before
  the first hyphen names the format, or for the evaluator inputs the
  part after `pairs-` (`name`, `wordtag`, `lemma`,
  `chunk`, `token`, `sentence`, `doccat`, `language`, `conllu`). They hold ordinary samples and the
  lines the streams treat unusually: untyped and nested start tags, an
  empty type, stray and missing end tags, empty names, tags that are not
  whole tokens, one-token samples, empty and whitespace-only lines, CR LF
  and lone CR, tokens without `_`, lines with too few or too many
  columns, two empty lines in a row, chunk tags that start phrases
  without `B-` or are one character long, `<SPLIT>` at the ends of a
  token and twice in a row, `<LF>` and `<CR>` escapes, a category with no
  text, lines without a language, tabs, no-break spaces, `U+0085`,
  `U+001C`, emoji and accented letters, and a last line without a line
  feed. The CoNLL-U files (Spanish, Portuguese and English sentences
  written for this project) hold multiword tokens with `_` among their
  words' values, an empty node, every comment `ConlluStream` reads, a
  sentence without a `text` comment and one of comments only, and the
  sentences it refuses or reads unusually: nine columns, an empty last
  column, a line of spaces, multiword ids with a leading zero, running
  backwards, past the words, past Java's `int`, out of order, repeated
  and overlapping, a bad `text_` language, forms missing from the text,
  a form found inside an inserted `<SPLIT>`, empty forms, CR LF and lone
  CR, and whitespace that only one whitespace mode splits on. The
  `pairs-` files hold a reference sample and then a prediction for it,
  pair after pair: predictions that are right, wrong, shorter and
  longer than the reference, chunk tags with no phrase, token and
  sentence spans past the reference's text (also after accented
  letters and emoji, whose UTF-16 offsets differ from their byte
  offsets), a name predicted twice,
  pairs with nothing to find, and sentence ends that only one
  whitespace mode trims.

  The `conll02`, `conll03` and `conllx` files, and `chunk-conll2000.txt`,
  are sentences written for this project in the layouts of the CoNLL
  shared tasks (no text of the CoNLL corpora): document starts that
  clear adaptive data, one followed by an empty line and one by a
  sentence, one at the end of the file, a line that starts with
  `-DOCSTART-` and goes on, IOB1 and IOB2 name tags, `I-` tags that open
  a name or change its type, adjacent `B-` names, unknown types, invalid
  tags, lines with too few and too many fields, a line of spaces, CR LF,
  accented letters, ten and fewer columns, empty columns, and a last line
  without a line feed.

  The `masc-` files are corpus directories bundled into one text (see
  "MASC goldens" in `analysis/goldens-tool/README.md`), with text and
  stand-off annotation written for this project in the layout of MASC
  3.0 (GrAF); they hold no MASC text. They hold documents with and
  without Penn tokens and named entities, the documents
  `MascDocumentStream` skips (each annotation file that does not parse,
  bad ids and anchors, a missing segmentation file, a document type
  declaration) and those that end it (a header that does not parse, has
  no annotation, no `loc`, no primary data or a missing file, no sentence
  regions, no quarks), overlapping, touching and empty sentence regions,
  quarks shared by Penn tokens, tokens across sentences, missing tags,
  empty tokens, named entities over one and several tokens, overlapping
  and untyped names, character references, element names in other cases
  and with prefixes, digits of other scripts in anchors, accented letters,
  a character outside the Basic Multilingual Plane, a byte order mark and
  malformed UTF-8 that the JDK's decoder replaces otherwise than the
  Unicode recommendation. `masc-opennlp.txt` is Apache OpenNLP's MASC
  test corpus (`opennlp-core/opennlp-formats/src/test/resources/opennlp/tools/formats/masc/`
  at the goldens' OpenNLP commit, Apache License 2.0), unchanged.

- `inputs/ner/`: 12 files in the named entity corpus formats OpenNLP
  reads, written for this project except as noted below; the part of
  the id before the first hyphen names the corpus (`germeval`,
  `bionlp`, `evalita`, `census`, `muc`, `brat`). They hold names of
  every type, types left out, nested GermEval names, unknown types and
  tags, lines with too few or too many columns, comment, abstract and
  document start lines, empty and whitespace-only lines, census names
  with `MC` prefixes and a line that ends the list, and accented
  letters. The `muc-` and `brat-` files are corpus directories bundled
  into one text (see "MASC goldens" in `analysis/goldens-tool/README.md`):
  MUC-6 style SGML documents with names, nested and unclosed tags, files
  in other cases and subdirectories, a file that gives no sample and
  errors after a document's first samples, and brat documents with
  text-bound annotations over several fragments, relations, events,
  attributes, notes, overlapping names, each error of the annotation
  file parsers, annotation configurations that fail, and a text with a
  character outside the Basic Multilingual Plane (`brat-utf16`). They
  hold no text of the MUC or brat corpora. `germeval-opennlp.txt` is
  four sentences of the GermEval 2014 named entity data (CC BY 4.0;
  D. Benikova, C. Biemann and M. Reznicek, "NoSta-D Named Entity
  Annotation for German: Guidelines and Dataset", LREC 2014) as Apache
  OpenNLP's test file `germeval2014.sample` holds them (Apache License
  2.0), unchanged after the comment lines that say so.

- `inputs/parse/`: four files of Penn Treebank parses, one per line,
  written for this project: English trees with function tags, gap
  indexes, `-NONE-` traces in subject and object extraction chains,
  possessives, every bracket escape, the types the English rules name
  and types they do not; Spanish trees with AnCora labels for the
  AnCora rules; lines OpenNLP reads leniently or fails on (an outer
  bracket without a type, an extra or a missing `)`, text outside the
  brackets, `=` and `-` in labels, empty brackets, blank and whitespace
  only lines); and tokens with accented letters, Chinese and Devanagari
  letters, a no-break space, `U+2028`, an ideographic space and an emoji
  outside the Basic Multilingual Plane, with CR line ends. They hold no
  text of the Penn Treebank or AnCora corpora.
- `inputs/head-rules/`: 13 head rule files written for this project:
  rules with each direction, a later rule for a type already read, a
  sign and tabs in a line, CR line ends, 60 types (more than
  `new HashMap<>(30)` and `new HashMap<>(60)` hold before they grow, so
  the order `serialize` writes them in changes), regular expression
  tags, an empty file, and each error the readers throw: a blank line,
  a line starting with `#`, a count that is not a number, one above
  1,002, one above the `int` range and one below 2, more and fewer tags
  than the count, and a missing direction.

`analysis/goldens-tool/README.md` ("Runs") lists which run reads which set.

## Fixtures

`fixtures/` holds the models and dictionaries that are not published
models. They are checked in, and `analysis/goldens-tool/src/recipe.rs`
lists each with its SHA-512; `generate` and `verify` refuse a directory
that differs.

| File | What it is |
|---|---|
| `en-abbrev.bin`, `en-plain.bin` | English sentence models trained by `analysis/sentdetect/tests/fixtures/src/TrainSentenceModel.java` (see the README there): `en-abbrev` has an abbreviation dictionary of 13 entries and `useTokenEnd=true`, `en-plain` has no dictionary and `useTokenEnd=false` |
| `en-plain-rust.bin` | the `.bin` the Rust trainer writes when it trains `en-plain`'s events again (`analysis/train/tests/roundtrip.rs`, which checks that the fixture is that file): the `sent.model` entry and the manifest of `en-plain.bin`, byte for byte, in an archive written here (entries stored, manifest first); the JVM loads it so the goldens of the two files can be compared |
| `en-plain-qn-rust.bin` | the `.bin` the Rust trainer writes when it trains `en-plain`'s events with `MAXENT_QN` (cutoff 1, the trainer's defaults; `analysis/train/tests/roundtrip.rs` checks that the fixture is that file and that its `sent.model` entry has the bytes the training golden `train/sentdetect-twopass-cutoff1-qn/train/sentences` recorded from the JVM's `QNTrainer`); the JVM loads it so the goldens show how it splits the fixture inputs |
| `en-plain-perceptron-rust.bin` | the same for `PERCEPTRON` (cutoff 1, the trainer's defaults), checked against the training golden `train/sentdetect-twopass-cutoff1-unsorted-perceptron/train/sentences` |
| `pos-fixture.bin` | a copy of `analysis/sequence/golden/models/pos-fixture.bin`, a part-of-speech model OpenNLP trained by GIS on 15 sentences with Penn Treebank tags, the default descriptor and the case-insensitive tag dictionary `tags.tagdict` (`the`, `Run`, `fell`, `Paris`; `analysis/sequence/README.md`) |
| `lemmatizer-dictionary.txt` | a `DictionaryLemmatizer` dictionary written for this project: 48 `word<tab>tag<tab>lemma` lines with Universal Dependencies tags for words of the `en` and `multi` inputs, among them entries with several lemmas (`leave#leaf`), a word listed twice with one tag (the last wins), a word with two tags, an empty lemma field (`#`), fields after the lemma and a trailing tab (both ignored), entries with a capital letter (never found, since a lookup lowercases the token) and non-ASCII words |
| `chunker-fixture.bin` | a copy of `analysis/sequence/golden/models/chunker-fixture.bin`, a chunker model OpenNLP trained by perceptron on 8 sentences with Penn Treebank tags and `B-`, `I-` and `O` chunk tags for `NP`, `VP`, `PP`, `ADJP` and `SBAR` (`analysis/sequence/golden/inputs/train-chunk.txt`, `analysis/sequence/README.md`); the only chunker model here, since the published chunker models are OpenNLP 1.5 models, which this project does not use |
| `tokenizer-abbreviations.bin` | a copy of `analysis/maxent/tests/fixtures/models/tokenizer.bin`, a tokenizer model OpenNLP trained on six sentences with the case-insensitive dictionary `Dr.`, `U.S.`, `e.g.` (`analysis/maxent/tests/fixtures/README.md`) |
| `english-dictionary.xml` | an abbreviation dictionary written for this project, added to the English tokenizer model |
| `latin-detokenizer.xml` | Apache OpenNLP's test resource of `DictionaryDetokenizerTest`, unchanged (license header kept) |
| `en-detokenizer.xml`, `fr-detokenizer.xml`, `pt-detokenizer.xml` | Apache OpenNLP's English, French and Portuguese detokenizer dictionaries, `opennlp-tools/lang/{en,fr,pt}/tokenizer/`, unchanged (license header kept) |
| `quotes-detokenizer.xml` | two `RIGHT_LEFT_MATCHING` entries, written for this project |
| `zh-unigram-fixture.txt` | a frequency lexicon written for this project: the eight words of OpenNLP's `UnigramSegmenterTest`, pairs of words whose counts have equal products (`研究` 6 `生命` 4 and `研究生` 8 `命` 3, `研究` 6 `生物` 12 and `研究生` 8 `物` 9, `人民` 1 `大会` 148 and `人民大` 2 `会` 74, `a` 3 `bc` 2 and `ab` 2 `c` 3), CR LF and lone CR line ends, an empty and a blank line, a tab, a no-break space and an ideographic space before the count, fullwidth digits and a `+` sign in a count, a leading ideographic space, a word outside the Basic Multilingual Plane, Latin words and a word listed twice |
| `zh-unigram-modes.txt` | a frequency lexicon written for this project whose words depend on the whitespace mode: `U+001C` ends a word in the legacy mode only and `U+0085` in the Unicode mode only |
| `jieba-dict-subset.txt` | the lines of jieba's `jieba/dict.txt` (github.com/fxsjy/jieba, commit `67fa2e36e72f69d9134b8a1037b83fbb070b9775`, MIT license, see below) whose word occurs in an input of the `zh`, `multi` or `edge` set, unchanged and in the order of `dict.txt`; made with the command below |
| `mecab-fixture-ja/` | a MeCab dictionary directory written for this project: two lexicon files (`A-nouns.csv`, `b-others.csv`) of 37 entries with IPADIC-style part-of-speech features, a 5 by 5 `matrix.def`, a `char.def` with overlapping categories, a category without grouping and supplementary ranges, and an `unk.def`. Its costs make ties of each kind the lattice tokenizer breaks, and one surface has a comma in it (quoted, as MeCab quotes it) |
| `mecab-fixture-ko/` | a MeCab dictionary directory written for this project: one lexicon file of 23 entries with mecab-ko-dic-style features, a 4 by 4 `matrix.def`, and a `char.def` and `unk.def` with Hangul, jamo and Hanja categories |
| `mecab-fixture-partial/` | a MeCab dictionary directory written for this project whose `unk.def` has no `DEFAULT` template, so a character that starts no lexicon word and is outside its `KANJI` category has no candidate |
| `all-minilm-l6-v2-vocab.txt` | the WordPiece vocabulary of `testdata/all-minilm-l6-v2/tokenizer.json` (sentence-transformers/all-MiniLM-L6-v2, Apache-2.0), one piece per line in id order with a line feed after each, written with `jq -r '.model.vocab \| to_entries \| sort_by(.value) \| .[].key'`; 30522 pieces, the uncased BERT vocabulary |
| `en-head-rules.txt` | Apache OpenNLP's English head rules (`opennlp-tools/lang/en/parser/en-head_rules` at the goldens' OpenNLP commit, the same bytes as the test resource `opennlp-core/opennlp-runtime/src/test/resources/opennlp/tools/parser/en_head_rules`; Apache License 2.0, rules after Collins and Magerman), unchanged |
| `es-head-rules.txt` | Apache OpenNLP's AnCora Spanish head rules (`opennlp-tools/lang/es/parser/es-head-rules` at the same commit, Apache License 2.0), unchanged; its tags double their backslashes, so OpenNLP never matches those |
| `es-regex-head-rules.txt` | AnCora head rules written for this project whose tags use classes, ranges, negation, `?`, `+`, `^`, `$` and escapes, for the patterns OpenNLP's file does not reach |

`jieba-dict-subset.txt` was written from the repository root with

```
for f in analysis/goldens/data/inputs/{zh,multi,edge}/*.txt; do cat "$f"; echo; done > /tmp/inputs.txt
ALL=/tmp/inputs.txt perl -CSD -ne 'BEGIN { local $/; open my $f, "<:encoding(UTF-8)", $ENV{ALL} or die; $t = <$f> }
  ($w) = /^(\S+)/; print if index($t, $w) >= 0' <jieba>/jieba/dict.txt \
  > analysis/goldens/data/fixtures/jieba-dict-subset.txt
```

and has to be written again, with the goldens, when an input of those
sets changes. jieba's license:

```
The MIT License (MIT)

Copyright (c) 2013 Sun Junyi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Dictionary notices

The goldens under `lattice/mecab-ipadic-2.7.0-20070801/` hold the
feature columns (part of speech and reading) of the IPADIC entries the
JVM chose for each input, and the dictionary itself is not checked in.
IPADIC's `COPYING` asks that any copy, whole or modified, carry its
notice, which follows as it is in `mecab-ipadic-2.7.0-20070801.tar.gz`:

```
Copyright 2000, 2001, 2002, 2003 Nara Institute of Science
and Technology.  All Rights Reserved.

Use, reproduction, and distribution of this software is permitted.
Any copy of this software, whether in its original form or modified,
must include both the above copyright notice and the following
paragraphs.

Nara Institute of Science and Technology (NAIST),
the copyright holders, disclaims all warranties with regard to this
software, including all implied warranties of merchantability and
fitness, in no event shall NAIST be liable for
any special, indirect or consequential damages or any damages
whatsoever resulting from loss of use, data or profits, whether in an
action of contract, negligence or other tortuous action, arising out
of or in connection with the use or performance of this software.

A large portion of the dictionary entries
originate from ICOT Free Software.  The following conditions for ICOT
Free Software applies to the current dictionary as well.

Each User may also freely distribute the Program, whether in its
original form or modified, to any third party or parties, PROVIDED
that the provisions of Section 3 ("NO WARRANTY") will ALWAYS appear
on, or be attached to, the Program, which is distributed substantially
in the same form as set out herein and that such intended
distribution, if actually made, will neither violate or otherwise
contravene any of the laws and regulations of the countries having
jurisdiction over the User or the intended distribution itself.

NO WARRANTY

The program was produced on an experimental basis in the course of the
research and development conducted during the project and is provided
to users as so produced on an experimental basis.  Accordingly, the
program is provided without any warranty whatsoever, whether express,
implied, statutory or otherwise.  The term "warranty" used herein
includes, but is not limited to, any warranty of the quality,
performance, merchantability and fitness for a particular purpose of
the program and the nonexistence of any infringement or violation of
any right of any third party.

Each user of the program will agree and understand, and be deemed to
have agreed and understood, that there is no warranty whatsoever for
the program and, accordingly, the entire risk arising from or
otherwise connected with the program is assumed by the user.

Therefore, neither ICOT, the copyright holder, or any other
organization that participated in or was otherwise related to the
development of the program and their respective officials, directors,
officers and other employees shall be held liable for any and all
damages, including, without limitation, general, special, incidental
and consequential damages, arising out of or otherwise in connection
with the use or inability to use the program or any product, material
or result produced or otherwise obtained by using the program,
regardless of whether they have been advised of, or otherwise had
knowledge of, the possibility of such damages at any time during the
project or thereafter.  Each user will be deemed to have agreed to the
foregoing by his or her commencement of use of the program.  The term
"use" as used herein includes, but is not limited to, the use,
modification, copying and distribution of the program and the
production of secondary products from the program.

In the case where the program, whether in its original form or
modified, was distributed or delivered to or received by a user from
any person, organization or entity other than ICOT, unless it makes or
grants independently of ICOT any specific warranty to the user in
writing, such person, organization or entity, will also be exempted
from and not be held liable to the user for any such damages as noted
above as far as the program is concerned.
```

The goldens under `lattice/mecab-ko-dic-2.1.1-20180720/` hold the
feature columns of the mecab-ko-dic entries the JVM chose. mecab-ko-dic
is under the Apache License 2.0 (its `COPYING`,
https://www.apache.org/licenses/LICENSE-2.0); its archive has no
`NOTICE` file. The dictionary itself is not checked in.

## Goldens

`<annotator>/<model>/<set>/<id>.golden` is what the OpenNLP annotator
`<annotator>` produced with model `<model>` (the model file's name
without `.bin`, `.xml` or `.txt`) on `inputs/<set>/<id>.txt`. A run with options
has `+` and their names after the model (`+legacy`, `+keep-new-lines`,
`+alphanumeric`, `+english-dictionary`, `+penn`, `+beam1`), and a run without a model names
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

## BPE goldens

`bpe/merges-<n>/bpe/<id>.golden` is what OpenNLP's `BPETokenizerTrainer`
learned from `inputs/bpe/<id>.txt` alone with `numMerges` `<n>` (40 and
400), and what `BPETokenizer` with those merges gave for the same text:
the merges in order (`turbo:bpe-merges`, each side a string, a lone
surrogate written `\uXXXX`) and the tokens (`opennlp:tokens`). The run
has no model file; `merges-<n>` names it. The trainer counts symbols in
UTF-16 units, so on `emoji` its merges hold halves of surrogate pairs;
its tokens are whole characters. `analysis/bpe-trainer/tests/opennlp.rs`
compares the Rust trainer and tokenizer with them (docs/bpe-trainer.md).

## Training goldens

`train/<source>-<indexer>-cutoff<n>[-unsorted]/<set>/<id>.golden` is what
OpenNLP's event streams, data indexers and GIS trainer did on
`inputs/<set>/<id>.txt`: the events a source made from the text, the
event checksum, the labels, counts and indexed events of the named
indexer with the run's cutoff and sorting, and the GIS trainer's log
likelihood and parameters after every iteration (docs/analysis.md,
"Training goldens"). The run has no model file; its settings name it.
`analysis/train/tests/goldens.rs` compares the Rust indexers with them.

`train-events/<run>/<set>/<id>.golden` is what the two event streams did
with factory options (an abbreviation dictionary, the Portuguese and Thai
end characters, the German alphanumeric pattern with the optimization
on, CoNLL-U input) on `inputs/<set>/<id>.txt`: the events, the
checksum, the exceptions the sample stream threw, and the manifest, zip
entries and dictionary entry of a model file built from the factory
(docs/analysis.md, "Training goldens").
`analysis/train-segmentation/tests/goldens.rs` compares with them.

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
`sentdetect`, `tokenize`, `postag`, `lemmatize`, `chunk` and `langdetect` that
names a model and has no `error` record, records every
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
