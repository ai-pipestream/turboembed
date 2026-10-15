# Language detection

`turbo-langdetect` (`analysis/langdetect/`) is a port of OpenNLP 3.x's
`LanguageDetectorME` onto the analysis API (docs/analysis.md). It reads
OpenNLP's language detector models unchanged, such as
`langdetect-183.bin` with its 103 languages, and finds the language of a
whole document. Per-span language (code switching) is not built.

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
(`inputs/langdetect/samples.txt`: nine languages, and lines in
capitals, with a tab or many spaces in the text, empty text, a backslash,
CR LF, no final newline, and lines the sample stream skips;
`inputs/langdetect/tree-bucket.txt`: six lines whose unigrams share a
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

The recipe that trains a model on Tatoeba and gates it against
`langdetect-183.bin` is in `analysis/train-recipes` (its README, "The
language detector recipe").

## Not ported

- `probingPredictLanguages`, which detects over growing chunks of the
  text and stops when the best language is stable: no golden covers it.
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
