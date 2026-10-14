# turbo-train-components

Training OpenNLP's components in Rust: the event streams
`SDEventStream`, `TokSpanEventStream`, `POSSampleEventStream` and
`LemmaSampleEventStream`, the factories that set them up, the model
files `SentenceDetectorME.train`, `TokenizerME.train`,
`POSTaggerME.train` and `LemmatizerME.train` return, with every manifest
entry and artifact the JVM writes, and the reproduction of the published
UD 1.3 models from their treebanks.

| Module | OpenNLP |
|---|---|
| `sentdetect` | `SentenceDetectorFactory` (with the language's end characters from the `Factory` classes), `SDEventStream`, `SentenceDetectorME.train`, `SentenceModel` |
| `tokenize` | `TokenizerFactory` (with the language's alphanumeric pattern), `TokSpanEventStream`, `TokenizerME.train`, `TokenizerModel` |
| `postag` | `POSTaggerFactory` (the default feature generator descriptor or a given one, a tag dictionary or none), `POSSampleEventStream` over `ConfigurablePOSContextGenerator`, `POSTaggerME.train`, `POSModel` with its `generator.featuregen` and `tags.tagdict` entries |
| `lemmatize` | `LemmatizerFactory`, `LemmatizerME.encodeLemmas`, `LemmaSampleEventStream` over `DefaultLemmatizerContextGenerator`, `LemmatizerME.train`, `LemmatizerModel` |
| `dictionary` | `Dictionary` as `new Dictionary(InputStream)` reads it and `Dictionary.serialize` writes it, through the JDK's `ToStream` with indentation |
| `tagdict` | `POSDictionary` as `POSDictionary.create` reads it (a case-insensitive one is lowercased into a second map in the first map's order) and `POSDictionary.serialize` writes it |
| `java_hash` | `String.hashCode`, `StringList.hashCode`, and the iteration order of a `java.util.HashMap` |
| `package` | The `factory` and `serializer-class-*` manifest entries and the zip entry order of `BaseModel.serialize` |
| `ud` | The recipes of the eighteen `opennlp-*-ud-*-{sentence,tokens,pos,lemmas}-1.3-2.5.4.bin` models and the comparison with the published files |

The samples come from `turbo-formats` (native and CoNLL-U readers), the
context generators from `turbo-sentdetect`, `turbo-tokenize`,
`turbo-sequence` (the tagger's) and `turbo-lemmatizer`, so that training
and inference share one implementation of each, and the trainer, model
writer and `.bin` packaging from `turbo-train`. This crate connects them,
as the `train` methods do. It is a crate of its own
because `turbo-formats` depends on `turbo-tokenize`: a training module
inside the annotator crates would close a cycle, and a module in
`turbo-formats` would make the sample readers depend on the trainer.

## What the JVM does that a reader may not expect

All of these are kept and tested (`src/`, `tests/goldens.rs`,
`tests/fixtures.rs`):

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
  a class that holds half of a surrogate pair; the JVM trains on it, a
  Rust string cannot hold it, and this crate refuses the sample
  (`Error::LemmaClassSurrogate`). The `train-events` goldens have two
  such samples, which `tests/goldens.rs` checks are the only events this
  crate does not make.
- The published UD 1.3 models were trained with 300 iterations, not the
  150 that `ud-train.conf` in `apache/opennlp-models` says (below).

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
`tests/goldens.rs` compares this crate with every one of them.
`tests/fixtures.rs` trains the fixture models `en-plain.bin`,
`en-abbrev.bin`, `tokenizer-abbreviations.bin` and `pos-fixture.bin`
again from their training texts and checks the manifest, the
dictionaries, the descriptor and the model entries byte for byte, and
trains `pos-fixture-rust.bin` and `lemma-fixture-rust.bin`, the fixtures
this crate wrote from `analysis/goldens/data/inputs/tagged/`, which the
golden generator loads in the JVM and tags with
(`analysis/postag/tests/golden.rs`, `analysis/lemmatizer/tests/golden.rs`
compare the Rust annotators with those goldens).

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
60 to 97 percent of them differing in their last bits. The JVM of the goldens
container, running the fork's trainer on the same events, writes exactly
the parameters this crate writes and shows the same residual against the
published file, so the residual is not between this port and the fork's
trainer. Its cause is not known: `GISTrainer` uses `StrictMath`
throughout and merges its threads' sums in a fixed order, so neither
platform `Math` intrinsics nor thread timing explain it. The next
experiment is to run the OpenNLP 2.5.4 command line itself in the
container on the same treebank and compare its parameters with both.
The test asserts agreement to within `1e-12` and reports which
models match byte for byte (none does); the model files it writes under
`TURBO_REPRODUCE_OUT` load in `turbo-maxent` and in OpenNLP.
