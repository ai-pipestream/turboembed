# Document analysis

The analysis crates hold a document's text and what annotators find in
it: views of the text, sentences, tokens, the embedder's subwords, tags,
entities, vectors and chunks. Each kind of result is a layer, and each
layer is its own Rust type, so reading one gives that type with its own
fields. Annotators declare the layer types they read and write, and the
compiler holds them to it.

It is the base the chunker is built on, and the base OpenNLP 3.x is to
be rewritten on, as add-ons: OpenNLP is the reference, and the Rust side
follows it on its own schedule. The crates are Rust only for now;
nothing here is in the C header or the gRPC server.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-analysis` | `analysis/core/` | The document, views, layers, keys, provenance, annotators, the analyzer and the registry |
| `turbo-analysis-layers` | `analysis/layers/` | The layers add-ons share (below) |
| `turbo-chunker` | `analysis/chunker/` | The chunker |

They need nothing beyond the standard library and each other. A shared
layer lives in `turbo-analysis-layers`, so a consumer of a layer never
depends on the add-on that produces it; an add-on's own layers live in a
small schema crate beside it, for the same reason.

## Offsets

Every offset is a UTF-8 byte offset into one text view, half-open, as a
`u32`: the unit the tokenizer's spans already use (docs/tokenizer.md).
Both ends of a span must be on character boundaries: a layer with a
span that cuts a character is refused when it is added, not found later
by a panic.
UTF-16 and code point offsets, for Java and the wire, are conversions
at the edge and are not built yet.

## Views

The source text never changes. A step that changes text (Unicode
normalization, dehyphenation) writes a new `View`, with an `Alignment`
to its parent: the edits, in text order on both sides. Outside the edits
the two texts are the same byte for byte. Mapping an offset either way
is a binary search over the edits. An offset inside an edit widens to
the whole edit (a start to its start, an end to its end), so a mapped
span covers everything the original covered.

A normalizer writes its text through a `ViewBuilder`, which records the
alignment as it goes, so no view exists without one. A layer's spans
name the view they are in, and `Document::to_source` maps them back
through any chain of views to exact source bytes.

## Layers

A layer implements `Layer`: an id, a schema version, what its items are
anchored to, how many there are, and a check of its own invariants. Its
data is stored by column (a `Vec` of starts, a `Vec` of tags), never as
an object per item. An item is anchored one of three ways:

- `Text`: each item carries a span of one view (sentences, tokens).
- `Aligned`: item i describes item i of another layer (a tag per
  token, a vector per sentence). The document checks the lengths match.
- `Document`: the layer describes the whole document (its language).

`Ix<L>` is an index into layer `L` and no other: an `Ix<Tokens>` used
as an index into sentences does not compile. It is stored as i + 1 in a
`NonZeroU32`, so an index that may be absent (a dependency head that may
be the root) is still four bytes.

Model probabilities and scores (`PosTags.prob`, `Entities.score`,
`Confidence<L>`, `Language.score`) are stored as f64. OpenNLP computes
them as Java doubles, and the parity tests compare them with the JVM's
output to within 1e-9, which f32 (about seven significant digits) cannot
hold. Vectors (`Vectors<L>`) and the chunker's costs (`Boundaries.cost`,
`Chunks.cost`) are f32: they are computed in Rust and never compared
with OpenNLP.

A layer is added once and never changed. Before it is taken the
document checks that the layer is not already there, that no other
layer type has its id, the layer's own invariants, that its spans are
in bounds and on character boundaries, that an aligned layer is as long
as its anchor, and that every index into another layer (a token's
sentence, an entity's tokens) is inside it. A document clones
cheaply, since its layers are shared, so branches of a pipeline can
grow copies and merge them; a merge where both branches wrote the same
layer, or both made a view, is refused and changes nothing.

## Keys

A `Key<L>` names one instance of layer `L` on a document. Most layers
have one, `Key::DEFAULT`; gold tokens beside predicted ones, or a
normalized view beside the source, are named instances
(`Key::named("gold")`). `KeyId` is a key with its type erased, for
pipelines built at run time; it prints as the layer's id and the name,
`opennlp:tokens@gold`.

A layer keeps the instances it was made on: an aligned layer's anchor,
and the layers it indexes. They are the instances its step read (for a
layer added by hand, those its provenance lists, or the default ones),
so tags made on the gold tokens say so (`Document::made_on`). A step
that would read such a layer beside another instance of the one it was
made on (those tags, with the predicted tokens) fails on that document
with `Error::Unaligned` instead of being handed items that do not line
up. Reading both instances, as an evaluation does, is fine.

One id names one layer type. A second type under an id is refused on
the document, whatever its instance name, and a pipeline holding both is
refused when it is built: on the wire they would be one layer.

## Provenance and status

Every layer records where it came from: the annotator and its version,
a hash of its configuration, the model (bundle id, revision and
sha256), where the model ran (backend, device, precision), and the
layers it read. An optional input is listed only when the document had
it.

A layer's status is kept apart from its data. A layer never written has
no status; one written with no items is `Complete` and empty; one its
annotator declined is `Skipped`, with the reason; one its annotator
failed to make is `Failed`, with the error. An annotator declines a
document by returning `Error::Skipped`. A step that needs a skipped or
failed layer is skipped in turn, and says why: `opennlp:language was
skipped: no language evidence`, or `opennlp:tokens failed: ...`. So
"skipped" is never mistaken for "found nothing", and a stored document
says why a layer is missing.

## Annotators

An `Annotator` declares what it reads and writes as tuples of layer
types, up to six of each. One that reads `(View, Tokens)` and writes
`(PosTags,)` is handed a `&View` and a `&Tokens` and must return a
`PosTags`. It is handed nothing else, not the document, so it cannot
read a layer it did not declare and its provenance is complete; text
comes the same way, through a declared `View`. An input it uses only
when it is there is written `Option<L>`, and arrives as `Option<&L>`.

`annotate` handles one document. `annotate_batch` handles a whole batch
at once: by default it calls `annotate` on each, and a neural annotator
overrides it to pack every document's input into one device run. An
annotator that returns `Error::Annotator` fails that document alone. A
panic does too: it is caught, and a batch that panics is run again one
document at a time to find the documents that cause it.

`Bound` binds an annotator to the instances it reads and writes:
the default ones, or named ones (`.read(0, "folded")`).

## The analyzer

`Analyzer::builder()` takes the steps in any order, and the layers the
documents arrive with (`given`), such as structure from the parser
upstream. `build` refuses a pipeline that cannot run, naming why: two
layer types under one id, a layer two steps write, a required input
nothing writes, an aligned output whose anchor the step does not read,
or a cycle. It then orders the steps by what they read and write into
stages.

`run` takes a batch of documents. Each stage runs over the whole batch
before the next starts, so a batching annotator sees every document at
once; steps in one stage compute side by side on threads, then store
their layers. A step that fails on a document fails only its own layers
there: the steps that need them are skipped and say why, the others
still run, and no other document is touched. Each document's result is
its first failure. `plan` lists the stages.

## The registry

A pipeline can also be assembled from configuration: a list of (name,
spec) lines, where a spec is a model location and options in order. Each
name is resolved through a `Registry<dyn Step>`, whose providers each
have a name, a priority, a cheap availability check and a claim on a
spec. The best available provider that supports the spec wins, and two
that tie at the top priority are an error, not a silent pick. Add-ons
are registered by the program that links them; nothing is found by
scanning. The shape is OpenNLP 3.x's `Provider` SPI (`Provider`,
`ProviderSpec`, `Providers`), and what a provider makes is the
registry's type parameter, as the service interface is in Java.

## Layer ids

A layer the OpenNLP gRPC standard defines has the standard's id (the
layer ids in `opennlp_document.proto`), so Java, Rust and the wire call
it one thing. A layer the standard lacks is `turbo:` until the standard
adds it. An add-on's own layers are in its own namespace. A layer
generic over another names both: `opennlp:embeddings<opennlp:sentences>`
is a vector per sentence, a different layer from
`opennlp:embeddings<turbo:chunks>`. A change to what a shared layer's
fields mean bumps its `SCHEMA`.

| Layer | Id | Anchored to | Crate |
|---|---|---|---|
| `View` | `turbo:view` | the document | `turbo-analysis` |
| `Structure` | `turbo:structure` | text | `turbo-analysis-layers` |
| `Sentences` | `opennlp:sentences` | text | `turbo-analysis-layers` |
| `Tokens` | `opennlp:tokens` | text | `turbo-analysis-layers` |
| `Subwords` | `opennlp:subwords` | text | `turbo-analysis-layers` |
| `PosTags` | `opennlp:pos` | tokens | `turbo-analysis-layers` |
| `Lemmas` | `opennlp:lemmas` | tokens | `turbo-analysis-layers` |
| `Dependencies` | `opennlp:dependencies` | tokens | `turbo-analysis-layers` |
| `Entities` | `opennlp:entities` | text | `turbo-analysis-layers` |
| `Boundaries` | `turbo:boundaries` | sentences | `turbo-analysis-layers` |
| `Chunks` | `turbo:chunks` | text | `turbo-analysis-layers` |
| `Vectors<L>` | `opennlp:embeddings<L>` | `L` | `turbo-analysis-layers` |
| `Confidence<L>` | `turbo:confidence<L>` | `L` | `turbo-analysis-layers` |
| `Language` | `opennlp:language` | the document | `turbo-analysis-layers` |

`Chunks` are retrieval chunks. The standard's `opennlp:chunks` are
phrase chunks (noun and verb groups), a different layer, which a phrase
chunker will write under that id.

## The chunker

`turbo-chunker` reads the sentences, the embedder's subwords, the
boundary costs and, when the document has it, its structure, and writes
the chunks. Every length is counted in the embedder's own tokens, over
each chunk's whole span: what lies between two of its sentences (a
paragraph break, which byte-level BPE makes a token of) is in the chunk
too, and a subword a chunk's edge cuts counts on both sides, as each
chunk's own tokenization makes a token of its part. `Subwords` holds
only the text's own pieces; the special tokens a template adds and a
prompt prefix are the budget's overhead.

It chooses every boundary of a run of sentences together, the way TeX
breaks a paragraph into lines, rather than filling each chunk to the
budget and cutting wherever it lands. A chunk costs the boundary that
closes it plus its squared distance from the target size, relative to
the target and weighted; the optimizer finds the cut of least total
cost. No chunk is longer than the budget's maximum, unless one sentence
alone is: that sentence is then a chunk of its own, marked as over the
budget, for a clause splitter to cut further. Time is the number of
sentences times the most that fit in one chunk.

What a boundary costs is the `Boundaries` layer: a cost and a reason
after each sentence. Each signal annotator (structure, topic shifts,
coreference, discourse markers) writes its own named instance, and one
step weighs and sums them into the default instance, since a layer has
one writer; the chunker reads only that.

With structure, no chunk crosses a top-level section. Each section is
its own run, so an edit re-chunks only its own section, and each
section becomes a chunk one level up, the parent of the chunks cut
from it. Every chunk records its level, its parent, its sentences, why
it starts where it does, what it cost the optimizer and its length in
model tokens.

## Generating golden files

The goldens in `analysis/goldens/data` are made by `turbo-goldens`
(crate `turbo-goldens-tool` in `analysis/goldens-tool`). It builds a
container from a pinned `eclipse-temurin:21-jdk` digest with OpenNLP
built from a pinned commit of the fork, runs the Java generator in it
without a network over the inputs and the models (each model checked
against its published SHA-512), and writes
`<annotator>/<model>/<set>/<id>.golden` for every annotator and input.
`turbo-goldens verify` regenerates them into a scratch directory and
fails on any file whose bytes differ. The first goldens (sentence
detection, tokenization and language detection) were made by running
the generator directly under the JDK named in their headers; the
container run is to be proven on a machine with Docker.
`analysis/goldens-tool/README.md` has the commands.

## Decision point dumps

Beside each golden, `<annotator>/<model>/<set>/<id>.dump` records every
decision the annotator asked its maxent model for on that input: the
context strings the context generator produced and the probability of
each outcome that `eval` returned. A Rust context generator and the
Rust maxent and perceptron evaluation can be checked against it one
decision at a time, before a whole annotator's golden lines up. The
generator writes them for `SentenceDetectorME`, `TokenizerME` and
`LanguageDetectorME`.

### The format

UTF-8 text, LF line ends, fields separated by one tab, and a line end at
the end of the file, as in a golden. The cells have the golden format's
types: `u32` (decimal, no sign, no leading zeros), `f64` (the 16
lowercase hex digits of the raw IEEE 754 bits, a space, then
`Double.toString`, for example `3fe0000000000000 0.5`) and `str`.

```
dump 1
generator	turbo-goldens-java	1
opennlp	<40 hex commit>	<maven version>
jdk	<java.version>	<java.vm.version>	<java.vendor>
unicode	<Unicode version>
input	<set>/<id>	<length in UTF-8 bytes>	<length in UTF-16 units>	<sha256 hex>
model	<file name>	<sha512 hex>
annotator	<fully qualified Java class>	<method name>
point	<index u32>	<position u32>	<contexts u32>	<outcomes u32>	<unit u32>	<values>
context	<str>
...
outcome	<label str>	<f64>
...
end	<index u32>
point	...
```

- The first line is exactly `dump 1`; the `1` is the format version.
- The header records are the ones the golden at the same path has, with
  the same values: the same `generator`, `opennlp`, `jdk`, `unicode` and
  `input` lines, one `model` line per model loaded, and the same
  `annotator` class and method.
- Then one block per `eval` call, in the order the annotator made them,
  numbered from 0. A dump with no decision points (a text without an
  end of sentence character) is its header alone.
- `position` is the UTF-8 byte offset into the input that the decision is
  about, or 0 for a decision about the whole document. For the sentence
  detector it is the offset of the end of sentence character the scanner
  found (the `position` argument of `SDContextGenerator.getContext`); for
  the tokenizer it is the offset of the character a split would go before,
  inside a whitespace separated token; for the language detector it is 0.
- `unit` is 0 when the UTF-16 index OpenNLP decided at starts a code
  point, and 1 when it lies between the two halves of a surrogate pair.
  The tokenizer walks UTF-16 units, so it asks about such positions inside
  an emoji or any other character outside the Basic Multilingual Plane;
  `position` is then the byte offset of that pair's code point. The two
  fields together name every decision exactly without a UTF-16 offset in
  the file.
- `contexts` lines follow, `context	<str>`, in the order the model received
  them. `values` is `none` when the annotator called `eval(String[])`, and
  `given` when it called `eval(String[], float[])`; each context line then
  has a third field, the context's feature value widened to `f64`.
- `outcomes` lines follow, `outcome	<label>	<f64>`, one per outcome of the
  model in its outcome index order (`getOutcome(0)`, `getOutcome(1)`,
  ...), with the probability `eval` returned for it.
- `end	<index>` closes the block; the index must match.

The `str` escapes are the golden format's: `\t`, `\n`, `\r`, `\\`, and
`\uXXXX` (four lowercase hex digits, one UTF-16 unit) for any other
character below U+0020. A surrogate that is not half of a pair is
written as `\uXXXX` as well, because it has no UTF-8 form. The language
detector's n-grams and the tokenizer's contexts are cut at UTF-16 units
and hold such halves (the language detector dump of `multi/mixed` and the
tokenizer dumps of `en/emoji` and `multi/mixed` have them); a reader that
decodes into Rust strings has to keep such a field as UTF-16 units or as
its escaped text. The generator writes goldens through the same escape,
and its UTF-8 encoder fails rather than writing a replacement character.

### How the decisions are captured

The generator builds a second instance of each annotator over a
recording model. `MaxentRecorder` wraps the loaded maxent model, forwards
every call to it and records each `eval` call: copies of the context
array, of the feature values when there are any, and of the returned
probabilities (the annotators and models reuse their buffers). It extends
OpenNLP's `AbstractModel` rather than only implementing `MaxentModel`,
because `TokenizerModel` and `LanguageDetectorModel` refuse a maxent model
that is not an `AbstractModel`. It is built over the loaded model's own
parameters, predicate map and outcome names, so the methods
`AbstractModel` makes final (`getBestOutcome`, `getAllOutcomes`,
`getOutcome`) answer exactly as the loaded model's would.

The recording models are made with the constructors that take a maxent
model: `SentenceModel(String, MaxentModel, Map, SentenceDetectorFactory)`,
`TokenizerModel(MaxentModel, Map, TokenizerFactory)` and
`LanguageDetectorModel(MaxentModel, Map, LanguageDetectorFactory)`. The
map is every manifest entry of the loaded model except `factory`, so the
language, versions, timestamp, training entries and factory settings are
the loaded model's; the generator checks that the recording model's
manifest equals the loaded one's in every entry but `factory`, and that
the loaded model's factory is exactly OpenNLP's default factory class.
A model constructor writes the factory's class name into the manifest
and makes its own factory instance from it through OpenNLP's
`ExtensionLoader`, so the recording factories are public classes in the
generator's package, which the generator registers with
`ExtensionLoader.registerAllowedPackage`.

How a decision's position is found, per annotator:

| Annotator | Approach | Why |
|---|---|---|
| `SentenceDetectorME.sentPosDetect` | The context generator is wrapped (`RecordingSentenceDetectorFactory`). Each `getContext(text, position)` call is reported to the recorder with the array it returned; the next `eval` must receive that very array, and the point takes its `position`. | The detector skips a candidate when it lies before the last accepted sentence start, which depends on earlier decisions, so the candidates cannot be listed from outside without re-implementing the loop. |
| `TokenizerME.tokenizePos` | The context generator is wrapped (`RecordingTokenizerFactory`) as above. It sees only one whitespace token (a new substring per token) and a UTF-16 index into it, so the requests of one token are grouped by that substring's identity, and each group is paired with the next span of `WhitespaceTokenizer` (without kept line breaks, the tokenizer's default) whose text equals the token. The position is the span's start plus the index. | After an accepted split at an abbreviation the tokenizer jumps over the abbreviation's dots, which depends on the decision, so the candidates cannot be listed from outside. Whether a whitespace token is evaluated at all depends on its text alone (its length and the alphanumeric pattern), so a skipped span can never have the text of an evaluated one; the generator checks this and fails otherwise. |
| `LanguageDetectorME.predictLanguages` | No context generator is wrapped. The detector makes exactly one `eval(String[], float[])` call per document (checked), and the point has position 0. | There is nothing to pair. The contexts are the distinct n-grams of the normalized text in the iteration order of the `HashMap` the detector counts them in (keyed by `CharBuffer`, whose hash depends only on the characters, so the order is the same on every run of one JDK), and every value is 1: the detector does not pass the counts (OPENNLP-1261 in its source). |

Nothing in a dump is inferred: every context, value and probability is
what `eval` received or returned, and every position is what the
annotator passed to its context generator. The layers the recording
annotator produces are written as a golden and compared byte for byte
with the plain annotator's golden for the same input; the generator
fails if they differ, so a dump is only written when recording changed
nothing the annotator computed.

### Making them

`Main` writes the dumps when given `--dumps` after its four arguments.
The driver always passes it (`GENERATOR_ARGS` in
`analysis/goldens-tool/src/recipe.rs`, forwarded by the container's
entrypoint), so `turbo-goldens generate` writes and installs goldens and
dumps together, and `turbo-goldens verify` compares both byte for byte.
`verify` and `generate` also check, at the header level only, that every
golden has a dump beside it, that every dump has a golden, and that each
dump's header parses and names the input and model of its path. The
checked in dumps were made by running the generator directly under the
JDK named in their headers, as the goldens were; the container run is to
be proven on a machine with Docker.

## Tests

`cargo test -p turbo-analysis -p turbo-analysis-layers -p turbo-chunker`
runs them on any machine:

- `analysis/core/tests/pipeline.rs` runs a pipeline of toy annotators
  over the shared layers: ordering by data, a normalized view whose
  tokens land on exact source bytes, a batch reaching a batching
  annotator in one call, failures, panics and skips per document, gold
  and predicted layers side by side and never read across, one type
  per id, and a pipeline from configuration.
- `analysis/chunker/tests/chunker.rs` chunks hand-made layers: a cut
  where the topic shifts instead of where the budget fills, sections
  that are never crossed, and a paragraph break's token counted inside
  a chunk. The chunker's unit tests hold the optimizer against filling
  each chunk to the budget.
- A doctest on `Ix` checks that an index into the wrong layer does not
  compile.

## Not built yet

- The annotators themselves: normalizers, sentence detection,
  tokenization, the subwords from turboembed's tokenizer, vectors
  through turboembed, and the boundary signals. The tests build their
  layers by hand or with toy annotators.
- UTF-16 and code point offsets at the edge.
- Golden layer files made on the JVM from a pinned OpenNLP 3.x commit,
  for the ports to be checked against.
- The evaluation harness that every change to the chunker is measured
  with.
- A gRPC surface, to be shared with the OpenNLP standard.
