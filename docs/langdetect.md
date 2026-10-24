# Language detection

`turbo-langdetect` (`analysis/langdetect/`) is a port of OpenNLP 3.x's
`LanguageDetectorME` onto the analysis API (docs/analysis.md). It reads
OpenNLP's language detector models unchanged, such as
`langdetect-183.bin` with its 103 languages, and finds the language of a
whole document. `turbo-langspan` (`analysis/langspan/`) builds language by
span (code switching) on it; see "Language by span" below.

It depends on `turbo-analysis`, `turbo-analysis-layers`, `turbo-maxent`
(the model file and its GIS classifier), `turbo-normalize` (the
normalizers OpenNLP runs first) and `turbo-unicode` (Java's lowercase
mapping and modified UTF-8).

## What it does

`LanguageModel::open` reads a model file and checks what OpenNLP checks:
the manifest's `Component-Name` is `LanguageDetectorME` and the archive
has a classifier in `langdetect.model`. A manifest without `factory`
gets OpenNLP's default `LanguageDetectorFactory`, as on the JVM. The
model's `ModelRef` is `opennlp:LanguageDetectorME:<language>` (`und` for
the published models), the OpenNLP version in the manifest as the
revision, and the SHA-256 of the model file.

`ContextGenerator` is `DefaultLanguageDetectorContextGenerator` with the
settings `LanguageDetectorFactory` gives it, and the counting step of
`LanguageDetectorME.predictLanguages`:

1. The text goes through the five normalizers the factory chains, in its
   order: characters outside the Basic Multilingual Plane, URLs and
   e-mail addresses, social media markup, digits, and repeated characters
   and whitespace (`turbo_normalize::language_detection::prepare`,
   described in docs/normalizers.md).
2. The result is lowercased one code point at a time with
   `Character.toLowerCase(int)` (`turbo_unicode::java::code_point::to_lower_case`,
   the simple one-to-one mapping), into UTF-16 code units.
3. Every run of 1, 2 and 3 code units is an n-gram: all n-grams of
   length 1 first, then 2, then 3, each length in text order. An n-gram
   can hold one half of a surrogate pair, though after step 1 no text
   holds a surrogate pair.
4. The JVM keeps each distinct n-gram once, as a key of a `HashMap` keyed
   by `CharBuffer`, lists the keys in the map's iteration order, puts
   them in that order into a second `HashMap` and hands that map's keys
   to the model in its iteration order, each with the value 1 (the
   counts are not passed; OPENNLP-1261 in the source). The generator
   gives the same n-grams in the same order. `CharBuffer.hashCode` is
   computed from the last code unit to the first, and the order of each
   map comes from a model of `java.util.HashMap`'s buckets: the bucket
   is the hash with its high half folded in, masked by the capacity; the
   capacity starts at 16 and doubles past three quarters full, or when a
   bucket reaches nine keys below a capacity of 64; a bucket keeps its
   keys in the order they were put, and iteration walks the buckets in
   index order.
5. Each n-gram is written as Java modified UTF-8 into one buffer, the
   form the model keys its predicates by, so a lone surrogate is the
   three bytes the JVM looks up.

The order matters because the GIS classifier adds up each n-gram's
parameters in the order it receives them, and floating point addition
is not associative: the same n-grams summed in another order give
probabilities that differ in the last bits.

`LanguageDetector::predict_languages` evaluates the model on the
context (`MaxentModel::eval_features`, with `StrictMath`'s `exp`) and
ranks every language of the model by probability, best first. Languages
of equal probability keep the model's outcome order, as `Arrays.sort`
(a stable sort) keeps them, and probabilities are ordered by
`Double.compare`. `predict_language` is the first language of that
ranking, found without sorting. `Ranking::has_evidence` says whether at
least one n-gram of the text is a predicate of the model; without one,
every probability is the model's prior and the ranking is the outcome
order.

`LanguageDetector::probing_predict_languages` and
`probing_predict_languages_with` (`ProbingConfig`, `ProbingContext`)
reproduce `LanguageDetectorME.probingPredictLanguages`:

1. The text is stepped through in chunks of at most
   `min(chunk_size, max_length - start)` Unicode code points
   (`CharSequence.codePoints().skip(start).limit(actualChunkSize)`),
   with `chunk_size = 200` and `max_length = 10_000` in
   `ProbingConfig::default()` (`LanguageDetectorConfig.DEFAULT_LANGUAGE_DETECTOR_CONFIG`),
   stopping when no code points remain before `max_length` or when
   `seenEnough` returns true.
2. Each chunk is normalized, lowercased and counted on its own
   (`ContextGenerator::generate_chunk`), and each n-gram not yet in the
   cumulative `ProbingContext` is inserted in the chunk's first `HashMap`
   bucket order (`updateCounts` into `ngramCounts`).
3. After each chunk, the model evaluates the cumulative map's keys in
   its current `HashMap` bucket order (each with weight 1.0) and ranks
   the outcomes as `predict_languages` does.
4. `seenEnough` keeps a sliding window of up to
   `min_consec_improvements + 1` top `(outcome, confidence)` pairs
   (`min_consec_improvements = 2` by default) and returns true once at
   least `min_consec_improvements` earlier chunks have been recorded,
   the margin `p[0] - p[1]` is at least `min_diff` (`0.20` by default,
   when `min_diff > 0.0`), and across the window the top outcome is
   unchanged and its confidence is non-decreasing (`!(last_conf > conf)`).
5. When the loop ends, `out` holds the last chunk's ranking (or the
   evaluation of the empty `ngramCounts` map when zero chunks were read,
   as on an empty text or `max_length == 0`), and the call returns the
   number of Unicode code points processed
   (`ProbingLanguageDetectionResult.getLength()`).

The buffers live in a `Scratch` the caller keeps, so a detector run over
many texts allocates for the n-grams and the probabilities only while
they grow. The normalizers return a new string for a text they change
(`prepare` borrows the text only when it is left as it is), and the
annotator allocates the layer it writes. The annotator keeps one
`Scratch` per thread.

## The annotator

`LanguageDetector` reads a `View` and writes `opennlp:language`
(`Language` in `turbo-analysis-layers`), anchored to the document: every
language of the model, best first, with its probability as an `f64`. A
text without evidence is declined with `Error::Skipped("no language
evidence")`, so a step that needs the language is skipped and says why.

## Parity with OpenNLP

The golden files under `analysis/goldens/data/langdetect/langdetect-183`
were made by `LanguageDetectorME.predictLanguages` on the JVM (the
OpenNLP commit and JDK are in their headers) for the 22 texts of the
`en` and `multi` input sets, with a decision point dump beside each:
the n-grams in the order the model received them and the probability of
each outcome (docs/analysis.md, "Decision point dumps").

`analysis/langdetect/tests/goldens.rs` checks:

- every n-gram of every dump (6221 in all), byte for byte and at its
  position, against the generator, without the model;
- every probability of every dump (22 texts times 103 languages), bit
  for bit, against `predict_languages`;
- every golden's `opennlp:language` layer, row by row, against what the
  annotator writes in a pipeline: the ranking exactly and every
  probability with `Tolerance::exact()`;
- `predict_language` against row 0 of every golden.

The golden files under `langdetect/langdetect-183+probing` (default
`LanguageDetectorConfig`: `chunkSize=200`, `maxLength=10000`,
`minConsecImprovements=2`, `minDiff=0.20`) and
`langdetect/langdetect-183+probing-chunk50` (`chunkSize=50`,
`maxLength=250`, `minConsecImprovements=2`, `minDiff=0.20`) were made by
`LanguageDetectorME.probingPredictLanguages` on the JVM over the 24 texts
of the `en`, `multi` and `langdetect` (`samples.txt` and
`tree-bucket.txt`) input sets. Each golden records `opennlp:language`
and `turbo:probing-length` (the code point count `getLength()` returned),
and each dump records one decision point per `predict(ngramCounts)`
evaluation. `analysis/langdetect/tests/goldens.rs` checks every chunk
context and outcome probability of every dump (100+ chunk evaluations and
over 25,000 n-grams in all), the final ranking and probabilities bit for
bit, and `turbo:probing-length` on every input.

The model is not checked in: `langdetect-183.bin` is fetched with
`cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models`,
which checks its published SHA-512. The tests look for it there, or in
the directory named by `TURBO_GOLDENS_MODELS`. Without it, the tests
that need it print where they looked and pass without comparing, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes them fail; the n-gram
comparison runs either way. CI does not fetch the model, so CI runs
the n-gram comparison and none of the probability comparisons.

`analysis/langdetect/tests/context.rs` checks the generator on texts
worked out by hand (the hash map order of `abab`, lowercasing, n-grams
across surrogate halves, U+0000, the n-gram lengths, the normalizers),
and `analysis/langdetect/tests/detector.rs` checks the detector, the
annotator and opening model files on a small GIS model written by the
test.

## Differences from OpenNLP

- The annotator declines a text without evidence. OpenNLP's
  `predictLanguages` returns every language at the prior probability,
  in outcome order, for an empty text, a text the normalizers empty
  (URLs, digits, emoji, whitespace) and a text none of whose n-grams the
  model knows; the first of them is then the model's first outcome
  (`tur` for `langdetect-183.bin`), which looks like an answer.
  `predict_languages` and `predict_language` return what OpenNLP
  returns, and say through `has_evidence` that there was none; the
  annotator returns `Error::Skipped` instead of writing that ranking.
  Tests: `the_annotator_writes_the_ranking_or_declines_without_evidence`
  (`analysis/langdetect/tests/detector.rs`) and
  `without_evidence_the_jvm_ranks_every_language_equal_and_the_annotator_declines`
  (`analysis/langdetect/tests/goldens.rs`).
- A `HashMap` bucket that reaches nine keys at a capacity of 64 or more
  becomes a red-black tree on the JVM. The tree's iteration order is
  reproduced (`turbo_jvm::hash_map`; a text of eleven distinct
  characters whose code units are all 33 modulo 64 makes one, test
  `a_bucket_that_java_turns_into_a_tree_comes_in_the_jvms_order` in
  `analysis/langdetect/tests/context.rs`), except between n-grams with
  equal hash codes: `CharBuffer` is not a class `HashMap` compares, so
  the JDK orders them by `System.identityHashCode`, which changes from
  run to run. `ContextGenerator::exact_order` returns false when that
  happens. The n-grams are the same, so the probabilities can differ
  from the JVM's only in the last bits. None of the 22 golden texts
  reaches a tree bucket.
- Lowercasing uses `turbo-unicode`'s Unicode 18.0 tables. JDK 21 has
  Unicode 15.0, so a capital letter added since then (such as U+A7CB,
  Unicode 16.0) is lowercased here and left as it is on the JVM, and its
  n-grams differ. docs/normalizers.md ("Differences from the JDK") lists
  the code points on which the tables differ. Test:
  `characters_new_since_unicode_15_lowercase_with_the_unicode_18_tables`
  (`analysis/langdetect/tests/context.rs`).
- A model whose manifest names a factory other than
  `opennlp.tools.langdetect.LanguageDetectorFactory` is refused when it
  is opened. OpenNLP would load the subclass, which can change the
  context generator; this port reproduces only the default one. Test:
  `other_components_and_factories_are_refused`
  (`analysis/langdetect/tests/detector.rs`).

`StringUtil.toLowerCaseCharBuffer` allocates as many code units as the
text has before lowercasing, so a mapping that changed the number of
code units would leave zeros at the end or overflow. No simple
lowercase mapping does, for any code point
(`no_simple_lowercase_mapping_changes_the_utf16_length`), so the
generator lowercases into a buffer of the same length and there is no
difference to record.

## Training

`turbo-train-components` (its `langdetect` module) trains a language
detector as `LanguageDetectorME.train` does (issue #429):
`LanguageDetectorFactory` with its default context generator; one event
per sample (`LanguageDetectorEventStream`), whose predicates are the
n-grams in the iteration order of the first hash map the generator fills
(`ContextGenerator::training_features`), where `predictLanguages` uses
the second map's order, which differs from it only when a bucket has
become a tree (test
`a_tree_bucket_can_give_training_and_prediction_different_orders` in
`analysis/langdetect/tests/context.rs`, with JDK 21's orders, and the
`tree-bucket` training goldens below); the `OnePass` indexer when the
parameters name none; the
event trainer the parameters name; and the model file `new
LanguageDetectorModel(model, entries, factory)` writes, with the
language `und` and the `langdetect.model` entry.
`default_training_parameters()` are those of
`ModelUtil.createDefaultTrainingParameters()` (`MAXENT`, 100 iterations,
cutoff 5), which the command line trainer uses when given none:
`langdetect-183.bin` records no parameters, so these are the ones it was
most likely trained with.

The goldens under `analysis/goldens/data/train-langdetect/` are the
JVM's training (goldens-tool `LanguageTrainGolden`) in two runs,
`default` (cutoff 5) and `cutoff1`, on samples written for them
(`analysis/goldens/data/inputs/langdetect/samples.txt`: nine languages, and lines in
capitals, with a tab or many spaces in the text, empty text, a backslash,
CR LF, no final newline, and lines the sample stream skips;
`analysis/goldens/data/inputs/langdetect/tree-bucket.txt`: six lines whose unigrams share a
bucket that becomes a tree, so training's n-gram order differs from
prediction's) and on the language format input.
`analysis/train-components/tests/langdetect.rs` compares the events,
their checksum, the index, the log likelihood and a digest of every
parameter after each GIS iteration, every weight after the first and the
last iteration bit for bit, the written model's SHA-256 and length, the
manifest and the zip entries. Bit for bit holds for events in the JVM's
predicate order; an event with equal-hash n-grams in a tree bucket is
not (`inexact_events`), and its weights can differ in the last bits. The
goldens hold no such event, which the test checks.

`analysis/goldens/data/fixtures/langdetect-fixture-rust.bin` is a model
trained here from the same samples (`analysis/train-components/tests/fixtures.rs`
checks that it is the file this crate writes). The goldens and dumps
under `langdetect/langdetect-fixture-rust` are OpenNLP running it on the
22 texts of the `langdetect-183` goldens (the 11 `en` and 11 `multi`
inputs), which shows that the JVM reads it, and
`analysis/langdetect/tests/fixture.rs` compares turbo-langdetect's
probabilities with them, each text's nine (one per language of the
fixture), bit for bit, and its rankings exactly (none of the 22 texts
reaches a tree bucket, where equal-hash n-grams could change the last
bits); the
annotator declines the two texts in which the model knows no n-gram
(Japanese and Chinese), where the JVM ranks every language equal.
The goldens and dumps under `langdetect/langdetect-fixture-rust+probing`
and `langdetect/langdetect-fixture-rust+probing-chunk50` run
`probingPredictLanguages` with the same fixture model over the 24 inputs
of `en`, `multi` and `langdetect`, and `analysis/langdetect/tests/fixture.rs`
checks every chunk context, every chunk and final probability bit for
bit, every ranking, and every stopping `length`.

The recipe that trains a model on Tatoeba and gates it against
`langdetect-183.bin` is in `analysis/train-recipes` (its README, "The
language detector recipe").

## Not ported

- Evaluation (`LanguageDetectorEvaluator`) and cross validation.

## OpenNLP findings

- `predictLanguages` hands the n-grams to the model in the iteration
  order of a `HashMap`, so its probabilities depend, in the last bits,
  on `HashMap`'s bucket layout, which a JDK may change.
- `predictLanguages` gives no sign that a text had no n-gram the model
  knows; such a text is ranked like any other, with the model's first
  outcome first.
- Training takes each event's predicates from the first of the two
  `HashMap`s, and `predictLanguages` from the second, so a model is
  trained on the n-grams in one order and run on them in another; the
  orders differ only when a bucket has become a tree.
- `langdetect-183.bin`'s manifest records no training parameters, only
  `TrainerType=Event` and the event hash, and its README points at a
  training script in a repository that could not be reached, so the
  parameters it was trained with are not on record.
  `ModelUtil.createDefaultTrainingParameters()`, which the command line
  trainer falls back to, sets no `TrainerType`; the trainer factory
  takes the algorithm for an event trainer.
- Its `Training-Eventhash` is an MD5 digest in hexadecimal
  (`fcf2dcf1bff70e8dc53ab4da7459bae5`, OpenNLP 1.8.3); OpenNLP 3.x writes
  the CRC-32C of `ChecksumEventStream` as a decimal number, so a model
  trained with 3.x cannot be compared with it by that field.

## Language by span

`turbo-langspan` (`analysis/langspan/`) finds the language of each part
of a document that changes language part way: a German e-mail quoting
English, a chat that switches between Spanish and English, a bilingual
notice. It uses the detector above for every score and adds no model.

### Layers

`opennlp:language` stays the document-level layer it is in OpenNLP, and
the spans are a new layer, `turbo:language-spans` (`LanguageSpans` in
`turbo-analysis-layers`). Giving `opennlp:language` spans instead would
change what every reader of it gets, and OpenNLP has no per-span layer
to follow. Each span has a language code (a tag in the layer's tag set:
the model's codes and `und`), the script most of its letters are in (a
`turbo_unicode::Script`) and a confidence. Spans are in text order and
touch: together they cover the text from the first word to the last.

The step (`SpanDetector`, annotator id `turbo-langspan:script-ngram`)
reads a `View` and, when the document has them, `opennlp:sentences` and
`opennlp:tokens`, and writes `turbo:language-spans` and
`opennlp:language`. Its `opennlp:language` is a summary of the spans:
every language of a span, with its share of the letters, largest first
(a tie goes to the language that comes first in the text). A pipeline
that wants OpenNLP's own ranking (every language of the model with its
probability) runs `LanguageDetector` instead; a document has one writer
of `opennlp:language`.

### How it works

1. Words: the letters of the text cut into words, each in one script
   group (Hiragana, Katakana and Bopomofo go with Han). Without a token
   layer a word ends at whitespace, at punctuation (except an
   apostrophe, hyphen, middle dot or full stop between two letters) and
   where the script group changes; with one, a token is at most one
   word, in the script group of most of its letters (the token
   `COVID対策` is one Latin word), so no span starts inside a token. A
   whitespace-separated chunk (or token) that the detector's
   normalizers remove, such as a URL, an e-mail address, a hashtag or a
   mention, makes no word.
2. Script first: consecutive words of one script group are a segment,
   and a segment can only be a language written in that script. The
   table of scripts per language (`turbo_langspan::scripts`) covers the
   103 languages of `langdetect-183.bin`; it comes from the writing
   systems of the languages, not from the model, whose weights give Han
   and Hangul letters small positive weights for nearly every language.
   A script with one language (Greek, Hebrew, Thai, Hangul, Georgian,
   Armenian and the Indic scripts other than Devanagari) needs no
   scoring and gets confidence 1; a script with none is `und` with
   confidence 0. A code the table does not know is allowed in every
   script.
3. Shortlist: the detector scores the whole segment and every clause
   (the words between two gaps with punctuation or a sentence break) of
   at least `clause_letters` letters (10); the best of the script's
   languages for each joins the shortlist. A segment with a shortlist of
   one is that language.
4. Labelling: each word, with a space either side, is scored against
   the shortlist (the natural log of the model's probability), and a
   Viterbi search finds the labelling with the best total less a cost
   for each switch: 0.20 between two words with only whitespace between
   them, 0.10 across punctuation, 0.05 across a sentence boundary or a
   line break. With a sentence layer its starts are the sentence
   boundaries; without one, sentence-final punctuation is.
5. Runs: each stretch of one label is scored again as a whole and given
   its best shortlisted language when that beats the label by more than
   `rescore_margin` (a natural log probability, 0 by default), and
   neighbours with the same language merge. A run's confidence is the model's probability of its language
   for the run's whole text, normalized over the languages of its
   script. The published model's probabilities are flat (a clear
   French sentence gets around 0.07 over the 65 Latin-script languages),
   so the confidence ranks runs against each other, is not calibrated,
   and should not be thresholded, rather than reading
   as a calibrated probability.
6. Spans: a run starts at a sentence start between the word before it
   and its first word, when there is one (snapped to the sentence layer);
   otherwise at its first word less the opening punctuation right before
   it (Ps and Pi characters, `¿`, `¡` and the straight quotation marks)
   when whitespace or the start of the text comes before that
   punctuation, moved back to the start of the token it falls in (snapped
   to the token layer). Sentence-final punctuation, closing brackets and
   quotation marks, and opening marks with no space before them stay with
   the run before: `休息。But` splits after `。`, and German `„Hallo!“Then`
   after `“`. The last run ends
   after its last word and the punctuation attached after it.

Steps 3 to 5 exist because the model scores single words badly: a short
word's probabilities favour a few languages (Waray and Minangkabau most)
whatever the word, so summing word scores over all the script's
languages labels most English and Spanish text as those. Ablation on the
development set: word scores over every Latin language alone get 18 of
30 documents exactly right; adding the whole-run rescoring gets 27; the
shortlist gets 30. To reproduce it: `cargo run --release -p turbo-langspan
--example evaluate -- <langdetect-183.bin>
analysis/langspan/tests/data/mixed-dev.txt` with `--no-shortlist
--no-rescore`, then `--no-shortlist`, then no flag.

Whole-run rescoring flips a run only when it beats the label by more
than `rescore_margin`. On the development set, with the shortlist,
every margin from 0 to infinite (no rescoring) gets all 30 documents
right, so the set does not choose one. Without the shortlist, a margin
of 1 gets 19 documents right against 27 for 0, so the default stays 0.
Admitting a clause's language to the shortlist only when it beats the
whole segment's language on the clause by a margin was tried too and
loses development documents at every margin tried (29 of 30 at 0.5, 10 at
2), so it is not used. A known miss neither fixes:
`Ich habe gestern (I was really tired) noch lange gearbeitet.` labels
the words after the parenthesis Frisian, because the clause
`noch lange gearbeitet.` scores Frisian above German and brings it onto
the shortlist (`short_review_texts_keep_their_language_and_punctuation`
in `analysis/langspan/tests/published.rs`).

### Fixtures and scores

`analysis/langspan/tests/data/` holds mixed-language documents with gold
spans, written for this crate: `mixed-dev.txt` (30 documents, 67 spans)
was used to choose the switch costs, `clause_letters` and
`rescore_margin`, and `mixed-test.txt` (40 documents, 87 spans,
including single-language documents, three-language documents and
one-word quotations) was not used to choose any setting. Nothing in the
repository shows when each file was written, and the held-out set has
been scored twice: once with the first settings, and once after the
review changes to span starts, tokens and rescoring, with the same
result. Together
they hold 28 languages in 11 scripts. A boundary counts as found when
no letter lies between it and the gold one (where the whitespace and
punctuation between two words go is convention), and "to the byte"
counts the predicted boundaries at exactly a gold boundary's byte, so a
boundary on the wrong side of punctuation shows; a span counts as right
when most of its letters get its gold language.

| Set | Detector | Documents exactly right | Span accuracy | Boundary precision | Boundary recall | Boundaries to the byte | Letter accuracy |
|---|---|---|---|---|---|---|---|
| dev | turbo-langspan, `langdetect-183.bin` | 30/30 | 67/67 | 37/37 | 37/37 | 37/37 | 1.000 |
| test | turbo-langspan, `langdetect-183.bin` | 33/40 | 81/87 (0.931) | 41/42 (0.976) | 41/47 (0.872) | 41/42 (0.976) | 0.956 |
| dev | lingua 1.8.0 `detect_multiple_languages_of` | 12/30 | 60/67 (0.896) | 26/47 (0.553) | 26/37 (0.703) | 26/47 (0.553) | 0.866 |
| test | lingua 1.8.0 `detect_multiple_languages_of` | 16/40 | 70/87 (0.805) | 27/56 (0.482) | 27/47 (0.574) | 27/56 (0.482) | 0.854 |

The development numbers are what the settings were chosen to get, not
a measure of accuracy. The seven test documents not exactly right: a
short English stretch next to Japanese or Hindi, and a French clause
of three words, read as Waray (the model's bias on few words, which a
run that short cannot outweigh); a one-word quotation
(`Schadenfreude`) and two short quotations, French inside English and
English inside Dutch, stay with the surrounding language; and
Indonesian reads as Sundanese. `analysis/langspan/tests/published.rs` records these scores
and fails when they change. The lingua rows come from a program outside
this repository, which is not checked in: it ran lingua 1.8.0 from
crates.io, restricted to the 68 languages it shares with the model (its
own option), and scored its output with this crate's `gold` module.
turbo-langspan used all 103 of the model's languages.

### Short texts against lingua

lingua's own test sets (single words, word pairs and sentences, up to
1000 of each per language, from the lingua-rs repository at commit
74eed3045bec53ba4feda2ac1780283f76fa302e) for the 68 languages lingua
and `langdetect-183.bin` share, with every detector restricted to those
68 languages. The model's two codes for Estonian (`est`, `ekk`),
Latvian (`lav`, `lvs`) and Persian (`fas`, `pes`) both count as right.
Accuracy is the mean over the languages of each language's share right.

| Set | OpenNLP document detector | turbo-langspan summary | lingua 1.8.0 (high accuracy) |
|---|---|---|---|
| single words | 0.617 | 0.629 | 0.760 |
| word pairs | 0.803 | 0.805 | 0.894 |
| sentences | 0.959 | 0.954 | 0.958 |

On whole sentences the span detector's best language is a little worse
than OpenNLP's ranking of the whole text, because a sentence it splits
in two can give the larger share to the wrong language. On single words
and word pairs the script filter helps a little; lingua's models are
better at short text than OpenNLP's. The weakest languages are the close
pairs for all three: Bosnian and Croatian with Serbian, Malay with
Indonesian. The test sets are not in this repository. The first two
columns come from `cargo run --release -p turbo-langspan --example
short_text -- <langdetect-183.bin> <lingua-rs>/language-models`; the
lingua column comes from a program outside this repository, not checked
in, that ran lingua's `detect_language_of` over the same lines. Lingua 1.7.2 resolves
to language model crates of the 1.8 format and then detects nothing,
so the comparison used lingua 1.8.0.

### Differences from the references

- lingua builds its candidate set from the whole text and every word of
  five or more characters, labels each word on its own and merges
  single-word results into their neighbours. Here the candidates come
  from the whole segment and from clauses, words are labelled jointly
  with a cost per switch that depends on the punctuation between them,
  and each run is scored again as a whole. Tests:
  `two_sentences_in_two_languages_are_two_spans_that_touch`,
  `a_quotation_takes_its_opening_mark_and_leaves_the_colon`
  (`analysis/langspan/tests/synthetic.rs`), the `decode` unit tests.
- whatlang and CLD3 filter by script before scoring; so does this, by a
  table of the model's languages instead of their per-script alphabets.
  A script the model has no language for is `und`, not a guess. Test:
  `scripts_with_one_language_need_no_scoring_and_scripts_with_none_are_und`.
- Han inside Hangul: Korean is written in Hangul only in the script
  table, so Hanja in a Korean text is a Han segment and gets a Han
  language (Chinese or Japanese), not Korean. Test:
  `han_inside_hangul_is_scored_as_a_han_language`.
- The `opennlp:language` this step writes lists only the languages of
  its spans, scored by share of letters, where OpenNLP's detector lists
  every language of the model with its probability. A text whose words
  are all in scripts the model has no language for gets an empty
  `opennlp:language` and `und` spans; a text with no words at all is
  declined, as `LanguageDetector` declines it. Test:
  `a_text_with_no_language_evidence_is_skipped_and_one_of_unknown_scripts_is_und`.

### Tests

`analysis/langspan/tests/synthetic.rs` runs the step on a model written
by the test (`x` is German, `y` English): spans and their summary,
quotations, snapping to sentences and tokens, single-language and
unknown scripts, declined texts, views and provenance. The unit tests
cover the script table, words, gaps, snapping, the Viterbi search and
the fixture format and scoring. `analysis/langspan/tests/published.rs`
checks that the script table covers every language of
`langdetect-183.bin` and records the fixture scores above; it needs the
model and skips without it, unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`,
which the CI step "parity against the JVM with the published models,
models required" sets.
