# turbo-train-components

Training OpenNLP's components in Rust: the event streams
`SDEventStream`, `TokSpanEventStream`, `POSSampleEventStream`,
`LemmaSampleEventStream`, `ChunkerEventStream`, `NameFinderEventStream`
and `LanguageDetectorEventStream`, the factories that set them up, the
model files `SentenceDetectorME.train`, `TokenizerME.train`,
`POSTaggerME.train`, `LemmatizerME.train`, `ChunkerME.train`,
`NameFinderME.train` and `LanguageDetectorME.train` return, with every
manifest entry and artifact the JVM writes, and the reproduction of the
published UD 1.3 models from their treebanks.

The lemmatizer's default factory is now native and case-preserving.
Use `LemmatizerFactory::opennlp` for the original event classes and JVM
model format. The published UD reproduction and golden tests select that
compatibility factory explicitly. See docs/lemma-case-evaluation.md.

| Module | OpenNLP |
|---|---|
| `sentdetect` | `SentenceDetectorFactory` (with the language's end characters from the `Factory` classes), `SDEventStream`, `SentenceDetectorME.train`, `SentenceModel` |
| `tokenize` | `TokenizerFactory` (with the language's alphanumeric pattern), `TokSpanEventStream`, `TokenizerME.train`, `TokenizerModel` |
| `postag` | `POSTaggerFactory` (the default feature generator descriptor or a given one, a tag dictionary or none), `POSSampleEventStream` over `ConfigurablePOSContextGenerator`, `POSTaggerME.train`, `POSModel` with its `generator.featuregen` and `tags.tagdict` entries |
| `lemmatize` | `LemmatizerFactory`, `LemmatizerME.encodeLemmas`, `LemmaSampleEventStream` over `DefaultLemmatizerContextGenerator`, `LemmatizerME.train`, `LemmatizerModel` |
| `chunk` | `ChunkerFactory`, `ChunkerEventStream` over `DefaultChunkerContextGenerator`, `ChunkerME.train` with `MAXENT`, `MAXENT_QN`, `PERCEPTRON` or `NAIVEBAYES`, `ChunkerModel` |
| `namefind` | `TokenNameFinderFactory` (the default descriptor or a given one without resources, the BIO or BILOU codec), `NameFinderEventStream` over `DefaultNameContextGenerator`, `NameFinderME.train` with an event trainer and its default parameters, `TokenNameFinderModel` with its `generator.featuregen` entry |
| `langdetect` | `LanguageDetectorFactory`, `LanguageDetectorEventStream` over `DefaultLanguageDetectorContextGenerator` (the n-grams in `getContext`'s order, which differs from `predictLanguages`' when a bucket became a tree), `LanguageDetectorME.train`, `LanguageDetectorModel` |
| `dictionary` | `Dictionary` as `new Dictionary(InputStream)` reads it and `Dictionary.serialize` writes it, through the JDK's `ToStream` with indentation |
| `tagdict` | `POSDictionary` as `POSDictionary.create` reads it (a case-insensitive one is lowercased into a second map in the first map's order) and `POSDictionary.serialize` writes it |
| `java_hash` | `String.hashCode`, `StringList.hashCode`, and the iteration order of a `java.util.HashMap` |
| `package` | The `factory` and `serializer-class-*` manifest entries and the zip entry order of `BaseModel.serialize` |
| `ud` | The recipes of the eighteen `opennlp-*-ud-*-{sentence,tokens,pos,lemmas}-1.3-2.5.4.bin` models and the comparison with the published files |

The samples come from `turbo-formats` (native and CoNLL-U readers), the
context generators from `turbo-sentdetect`, `turbo-tokenize`,
`turbo-sequence` (the tagger's and the chunker's), `turbo-lemmatizer` and
`turbo-langdetect`, so that training
and inference share one implementation of each, and the trainer, model
writer and `.bin` packaging from `turbo-train`. This crate connects them,
as the `train` methods do. It is a crate of its own
because `turbo-formats` depends on `turbo-tokenize`: a training module
inside the annotator crates would close a cycle, and a module in
`turbo-formats` would make the sample readers depend on the trainer.

## What the JVM does that a reader may not expect

All of these are kept and tested (`src/`, `tests/goldens.rs`,
`tests/namefind.rs`, `tests/fixtures.rs`):

- A sentence with no end character gives no event; a sentence's last
  end character is its split whatever follows it, and the others are
  non-splits.
- The factories resolve a missing end character set or alphanumeric
  pattern to the language's before anything reads them, so a trained
  model's manifest always has `eosCharacters` and `alphaNumericPattern`.
  An empty end character array is not resolved: the JVM then trains with
  the language's scanner and context generator (Thai's
  `SentenceContextGenerator`) but records `eosCharacters=` empty, so the
  model file does not describe the model; this crate refuses an empty set
  (`Error::EmptyEos`).
- The tokenizer's events are at UTF-16 positions, so a character outside
  the Basic Multilingual Plane gives an event between its two halves,
  with predicates that hold half a character; the predicates are
  `JavaString`s for that reason.
- A whitespace unit of one UTF-16 unit, and with the optimization on an
  alphanumeric one, gives no event at all.
- A sample token that overlaps a whitespace unit without lying inside it
  is skipped with a logged warning, and the search for tokens never goes
  back, so tokens must be in text order.
- An empty training token (CoNLL-U with an empty word form gives one)
  makes a split event at its position; at index 0 of a unit the context
  has `p1=bok` and no `p2` or `p1f1`, a context the tokenizer itself
  never asks for.
- `Dictionary.serialize` writes the entries in the order of a
  `HashSet<StringList>` (hash of the lowercased tokens joined with
  commas, table of 16 doubling at 13, 25, 49 entries), and
  `BaseModel.serialize` writes the zip entries in the order of a
  `HashMap<String, Object>`; both orders are simulated, so a sentence
  model has `manifest.properties`, `sent.model`,
  `abbreviations.dictionary` and a tokenizer model `manifest.properties`,
  `abbreviations.dictionary`, `token.model`.
- `new Dictionary(InputStream)` puts the entries before it reads the
  `case_sensitive` attribute, so entries that differ only in case
  collapse to the first even in a case-sensitive dictionary.
- The JDK's XML serializer writes a token's character outside the Basic
  Multilingual Plane as a decimal character reference (`&#128512;`), and
  U+007F and the C1 controls U+0080 to U+009F the same way (`&#127;`,
  `&#133;`); it escapes `&`, `<`, `>` and a carriage return (`&#13;`),
  writes `"` and `'` as they are, and throws on any other C0 control
  character, so a dictionary with such a token cannot be written into a
  model file (`Error::InvalidXmlCharacter`).
- `BaseModel.serialize` adds `serializer-class-abbreviations.dictionary`
  to the manifest only when a dictionary is present, and
  `serializer-class-tags.tagdict` only when a tag dictionary is.
- `POSTaggerFactory` given no descriptor loads `pos-default-features.xml`
  when its context generator is first asked for, so every trained
  `POSModel` carries a `generator.featuregen` entry (the published UD
  models do), and training uses no feature cache.
- `POSDictionary.create` puts the entries as written into a `HashMap`
  and, when the dictionary is case insensitive, copies them lowercased
  (`StringUtil.toLowerCase`, per code point) into a second map in the
  first one's iteration order; `serialize` writes the second map's order,
  so the `tags.tagdict` entry of a trained model is not the file it was
  read from (`pos-fixture.bin` shows this). A later `put` of a word
  already present replaces its tags and keeps its place.
- `POSModel` refuses a tag dictionary with a tag the model has no
  outcome for (`Error::UnknownDictionaryTags`, in the order of a
  `HashSet` of the tags, as the JVM's message lists them).
- With a tag dictionary the zip entries are `tags.tagdict`,
  `manifest.properties`, `generator.featuregen`, `pos.model`: the hash
  order puts the dictionary before the manifest.
- The lemmatizer's features name a missing suffix or prefix `suf=null`
  and `pre=null`, Java's string of a null reference, and an empty edit
  script is the class `_`.
- A lemma class is the shortest edit script over UTF-16 units, so a token
  or lemma with a character outside the Basic Multilingual Plane can give
  a class that holds half of a surrogate pair. A class is a `JavaString`
  for that reason, so such a sample trains, and the model file holds the
  class's modified UTF-8 as `writeUTF` writes it. The `train-events`
  goldens have four such classes in three samples (one each in
  `formats/lemma-basic` and `formats/conllu-whitespace`, two in one
  sentence of `tagged/lemmas`, the Rust lemmatizer fixture's training
  data), which `tests/goldens.rs` counts; their events and checksums are
  the JVM's. No golden holds a lemmatizer model the JVM trained on them:
  the written bytes are checked by the writer's unit tests, and
  `analysis/lemmatizer/tests/golden.rs` checks the JVM's lemmas and dumps
  over the Rust fixture, which the JVM reads.
- The published UD 1.3 models were trained with 300 iterations, not the
  150 that `ud-train.conf` in `apache/opennlp-models` says (below).
- `NameFinderME.train` puts `Algorithm=MAXENT`, `Cutoff=0` and
  `Iterations=300` into the caller's parameters where they are absent
  (`putIfAbsent`), so a name finder trained with no parameters file uses
  cutoff 0 and 300 iterations, not `TrainingParameters.defaultParams()`'s
  cutoff 5 and 100 iterations. The beam size is the `BeamSize` parameter,
  3 by default, and goes into the manifest.
- `NameFinderEventStream` adds an `AdditionalContextFeatureGenerator` in
  a window of eight tokens each side to the factory's generators, gives
  a name without a type the type `default` and, with a type given to the
  stream, gives every name that type. The previous-map feature generator
  remembers each token's outcome until a sample clears the adaptive data.
- The codecs index the outcome array directly: an empty name marks its
  first token's place as a start with BIO, and the token before it as a
  unit with BILOU, over whatever that token had. At the end of a
  sentence with BIO, or at its start with BILOU, that place is outside
  the array and the JVM throws `ArrayIndexOutOfBoundsException` (for a
  line holding only an empty name, "Index 0 out of bounds for length 0"
  with BIO and "Index -1 out of bounds for length 0" with BILOU; for
  `a <START:x> <END>`, "Index 1 out of bounds for length 1" with BIO),
  which ends training (`Error::UnencodableName`). The goldens of
  `name-edge` hold all three.
- `TokenNameFinderModel` refuses outcomes its codec's
  `areOutcomesCompatible` rejects ("Model not compatible with name
  finder!", `Error::IncompatibleOutcomes`). With BILOU an empty name right
  after a name overwrites that name's last outcome with a unit, so the
  type has a start and no last, and when no other name of the type ends
  normally the model is refused. Samples without a name give only
  `other`, which the trainer refuses before that ("Training data must
  contain more than one outcome").
- The trainers put `sort` into the caller's parameters (`true`, `false`
  for the perceptron) before they read the events, so the parameters
  after `NameFinderME.train` hold it even when training failed; this
  crate's trainers put it into a copy.
- `PERCEPTRON_SEQUENCE`, which trains from `NameSampleSequenceStream`, is
  not ported (`Error::NameSequenceTraining`), and neither are factory
  resources (word clusters, brown clusters, dictionaries): a descriptor
  that names one is refused.
- `ChunkerME.train` takes only event trainers: the sequence perceptron
  (`PERCEPTRON_SEQUENCE`), which the tagger and the name finder train
  with, is refused with `IllegalArgumentException("Trainer type is not
  supported: EVENT_MODEL_SEQUENCE_TRAINER")` (`Error::UnsupportedTrainerType`).
  The chunker model file has `chunker.model` before
  `manifest.properties`, the order of the JVM's artifact map.

## JVM reference

`analysis/goldens/data/train/` holds the events `SDEventStream` and
`TokSpanEventStream` made with default factories (`TrainGolden`), and
`analysis/goldens/data/train-events/` the events with a dictionary, the
Portuguese and Thai end characters, the German alphanumeric pattern with
the optimization on, and CoNLL-U input, plus what the sample stream
threw, the manifest and zip entries of a model file built from each
factory, and `Dictionary.serialize` of dictionaries built in the
generator (`EventGolden`), and the events of `POSSampleEventStream` with
and without a tag dictionary and of `LemmaSampleEventStream`, over word
and tag lines, word, tag and lemma lines and CoNLL-U in both tag sets,
with their model files, the `tags.tagdict` entry and
`POSDictionary.serialize` of dictionaries built in the generator
(`TaggerEventGolden`; docs/analysis.md, "Training the components").
`tests/goldens.rs` compares this crate with every one of them but the
chunker's. The `chunker-*` runs there (`ChunkerTrainGolden`) are
`ChunkerME.train` itself over the chunk inputs of the `formats` and
`tagged` sets: with OpenNLP's default parameters (GIS, 100 iterations,
cutoff 5), the perceptron with cutoff 1 and a beam size of 5, `MAXENT_QN`
with cutoff 1, and the sequence perceptron, which it refuses. Each
holds the events, what the sample stream threw, the event checksum, and
what training threw or the model it trained: its outcomes and
predicates in order, every parameter, the `chunker.model` entry's length
and SHA-256, the manifest and the zip entries. `tests/chunker.rs` trains
on the same samples with the same parameters and compares all of it:
every parameter must be the JVM's double, bit for bit (1,347 for the
defaults, 94,545 each for the perceptron and `MAXENT_QN`), and the
`chunker.model` entry's SHA-256 the JVM's, so the entries are the JVM's
bytes.
Beside the dictionary and pattern name finder runs,
`analysis/goldens/data/namefind/` holds the events of
`NameFinderEventStream` with the BIO and BILOU codecs, with a type given
to every name, and the names, the parameters after training, the
manifest, the zip entries, the digest of `nameFinder.model` and every
weight of the model `NameFinderME.train` trained over them with `MAXENT`,
`PERCEPTRON` and `MAXENT_QN` (`NameEventGolden`), and the names and
probabilities the JVM's `NameFinderME` finds with `ner-fixture-rust.bin`
(`NameFinderFixtureGolden`). `tests/namefind.rs` checks the events, the stream
errors, the parameters and the trained files against them: the
`nameFinder.model` entries are the JVM's byte for byte, so every weight
is equal, well within the 1e-9 the issue asks for; and that
`turbo-sequence`'s `NameFinder` finds the JVM's names with the JVM's
probabilities, bit for bit, over the Rust fixture.
`tests/fixtures.rs` trains the fixture models `en-plain.bin`,
`en-abbrev.bin`, `tokenizer-abbreviations.bin`, `pos-fixture.bin` and
`chunker-fixture.bin` again from their training texts and checks the
manifest, the dictionaries, the descriptor and the model entries byte
for byte, trains `turbo-sequence`'s name finder fixtures
`ner-bilou-fixture.bin` (GIS, with its descriptor and the BILOU codec)
and `ner-qn-fixture.bin` (`MAXENT_QN`, the default descriptor) again
from their training text, and trains `pos-fixture-rust.bin`,
`lemma-fixture-rust.bin`, `chunker-fixture-rust.bin` and
`ner-fixture-rust.bin`, the fixtures this crate wrote from
`analysis/goldens/data/inputs/tagged/`, which the golden generator loads
in the JVM and tags, lemmatizes, chunks or finds names with
(`analysis/postag/tests/golden.rs`, `analysis/lemmatizer/tests/golden.rs`
and `analysis/phrase-chunker/tests/golden.rs` compare the Rust
annotators with those goldens).
`tests/langdetect.rs` compares the language detector's training with the
`train-langdetect` goldens (the events, the index, every GIS iteration
and the model file; docs/langdetect.md, "Training"), among them
`inputs/langdetect/tree-bucket.txt`, whose events hold their n-grams in
`getContext`'s order where it differs from `predictLanguages`'; and
`tests/fixtures.rs` also trains `langdetect-fixture-rust.bin` from
`inputs/langdetect/samples.txt`, which the JVM loads and detects with
(`analysis/langdetect/tests/fixture.rs`).

`tests/cross_validate.rs` compares `TokenNameFinderCrossValidator`
(`namefind::cross_validate`) and `LanguageDetectorCrossValidator`
(`langdetect::cross_validate`) with the `cross-validate` goldens: per
fold the sides, the training events' checksum, the counts and every
prediction, and the merged precision, recall and F-measure or accuracy
(`jvm_accuracy`, the JVM's rounding; `accuracy` is the exact share).

## Reproducing the UD 1.3 models

`ud::RECIPES` are the eighteen published models
(`opennlp-{en,de,es,fr,pt}-ud-{ewt,gsd}-{sentence,tokens,pos,lemmas}-1.3-2.5.4`;
no sentence model was published for Spanish and French) as typed data:
the treebank (repository, train file, size and digests at Universal
Dependencies release 2.16, licence), the component, the factory and the
training parameters. The treebanks are CC BY-SA 4.0, so they are fetched
through `turbo-zoo` by digest after the licence is accepted, into the zoo
cache, and never committed; `ud::datasets()` is the zoo index of the
five. A published model that is not under `TURBO_GOLDENS_MODELS` (the
goldens use only the English and German taggers and lemmatizers) is
fetched through the zoo's built-in index (`ud::fetch_published_model`).

```text
cargo run -p turbo-zoo -- accept CC-BY-SA-4.0 --yes
cargo run -p turbo-goldens-tool -- fetch-models <dir>
TURBO_GOLDENS_MODELS=<dir> cargo test -p turbo-train-components --release --test reproduce -- --ignored --nocapture
```

The test is ignored by default and prints a report per model: the event
checksum, the model entry's digest and first differing byte, how many
parameters differ and by how much, and whether the manifest and the zip
entries agree. The settings the models were trained with come from
`ud-train.sh` and `ud-train.conf` in `apache/opennlp-models` and the
2.5.4 trainer tools' defaults, except the iteration count: the published
English sentence model's parameters are those of a run of 300
iterations, not the committed 150 (every iteration of a longer run was
compared with the file; 300 agrees to within a few thousand units in the
last place, 299 and 301 do not). With 300 iterations the events, the
manifest, the predicates and the outcome patterns of a published model
come out exactly for all eighteen models, and the parameters to within
2.2e-15 (German sentences) to 9.8e-15 (French tokens) for the sentence
and token models, 1.6e-14 (English) to 1.4e-13 (Spanish) for the taggers
and 8.0e-15 (English) to 1.8e-14 (Portuguese) for the lemmatizers, with
60 to 97 percent of them differing in their last bits. For the eight
sentence and token models, the JVM of the goldens container, running the
fork's trainer on the same events, writes exactly the parameters this
crate writes and shows the same residual against the published file
(checked in #394), so for them the residual is not between this port and
the fork's trainer; the taggers and lemmatizers have not been run through
the fork's trainer in the container, and their residual is only known to
be of the same size. Its cause is not known: `GISTrainer` uses `StrictMath`
throughout and merges its threads' sums in a fixed order, so neither
platform `Math` intrinsics nor thread timing explain it. The next
experiment is to run the OpenNLP 2.5.4 command line itself in the
container on the same treebank and compare its parameters with both.
The test asserts agreement to within `1e-12` and reports which
models match byte for byte (none does); the model files it writes under
`TURBO_REPRODUCE_OUT` load in `turbo-maxent` and in OpenNLP.
