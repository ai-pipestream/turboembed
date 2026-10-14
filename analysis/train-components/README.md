# turbo-train-components

Training OpenNLP's sentence detector and tokenizer in Rust: the event
streams `SDEventStream` and `TokSpanEventStream`, the factories that set
them up, the model files `SentenceDetectorME.train` and
`TokenizerME.train` return, with every manifest entry and artifact the
JVM writes, and the reproduction of the published UD 1.3 sentence and
token models from their treebanks.

| Module | OpenNLP |
|---|---|
| `sentdetect` | `SentenceDetectorFactory` (with the language's end characters from the `Factory` classes), `SDEventStream`, `SentenceDetectorME.train`, `SentenceModel` |
| `tokenize` | `TokenizerFactory` (with the language's alphanumeric pattern), `TokSpanEventStream`, `TokenizerME.train`, `TokenizerModel` |
| `dictionary` | `Dictionary` as `new Dictionary(InputStream)` reads it and `Dictionary.serialize` writes it, through the JDK's `ToStream` with indentation |
| `java_hash` | `String.hashCode`, `StringList.hashCode`, and the iteration order of a `java.util.HashMap` |
| `package` | The `factory` and `serializer-class-*` manifest entries and the zip entry order of `BaseModel.serialize` |
| `ud` | The recipes of the eight `opennlp-*-ud-*-{sentence,tokens}-1.3-2.5.4.bin` models and the comparison with the published files |

The samples come from `turbo-formats` (native and CoNLL-U readers), the
context generators from `turbo-sentdetect` and `turbo-tokenize`, and the
trainer, model writer and `.bin` packaging from `turbo-train`. This crate
connects them, as the two `train` methods do. It is a crate of its own
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
  to the manifest only when a dictionary is present.
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
generator (`EventGolden`; docs/analysis.md, "Training the sentence
detector and tokenizer"). `tests/goldens.rs` compares this crate with
every one of them. `tests/fixtures.rs` trains the fixture models
`en-plain.bin`, `en-abbrev.bin` and `tokenizer-abbreviations.bin` again
from their training texts and checks the manifest, the dictionary and the
model entries byte for byte.

## Reproducing the UD 1.3 models

`ud::RECIPES` are the eight published models (`opennlp-{en,de,es,fr,pt}-ud-{ewt,gsd}-{sentence,tokens}-1.3-2.5.4`)
as typed data: the treebank (repository, train file, size and digests at
Universal Dependencies release 2.16, licence), the component, the
factory and the training parameters. The treebanks are CC BY-SA 4.0, so
they are fetched through `turbo-zoo` by digest after the licence is
accepted, into the zoo cache, and never committed; `ud::datasets()` is
the zoo index of the five.

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
come out exactly for all eight models, and the parameters to within
2.2e-15 (German sentences) to 9.8e-15 (French tokens), with 60 to 85
percent of them differing in their last bits. The JVM of the goldens
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
