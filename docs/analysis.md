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
| `turbo-analysis` | `analysis/core/` | The document, views, offset tables, layers, keys, provenance, annotators, the analyzer and the registry |
| `turbo-analysis-layers` | `analysis/layers/` | The layers add-ons share (below) |
| `turbo-chunker` | `analysis/chunker/` | The chunker |
| `turbo-jvm` | `analysis/jvm/` | JVM behaviour the OpenNLP ports must reproduce bit for bit |
| `turbo-goldens` | `analysis/goldens/` | Golden layer files from the JVM: the reader, the writer, the comparison, SHA-256 and SHA-512 |
| `turbo-eval` | `analysis/eval/` | The evaluation harness: measures, significance tests, records and the gate (Evaluation, below) |

They need nothing beyond the standard library and each other, except
`turbo-eval`, which uses serde for its records and the core for
turbo-bench's provenance types. A shared layer lives in
`turbo-analysis-layers`, so a consumer of a layer never depends on the
add-on that produces it; an add-on's own layers live in a small schema
crate beside it, for the same reason.

## JVM parity

Some of OpenNLP's results depend on the Java platform rather than on
OpenNLP, and `turbo-jvm` reproduces those parts exactly, so a port can
match the JVM bit for bit:

- `chars`: Java sees text as UTF-16 code units. `Utf16Units` yields the
  units of a `&str` with the UTF-8 byte offset of each unit's character,
  for context generators, `StringPattern` and character n-grams;
  `utf16_len` is `String.length()`; `compare_to` is `String.compareTo`,
  which orders by code unit and so differs from `str::cmp` for
  characters outside the Basic Multilingual Plane; `is_java_whitespace`
  is `Character.isWhitespace`.
- `random`: `JavaRandom` is `java.util.Random` from a seed, and its
  `shuffle` is `Collections.shuffle(List, Random)`, as OpenNLP's readers,
  cross-validation and trainers use them.
- `strict_math`: `exp` and `log` port fdlibm, which `StrictMath` computes
  and which the maxent, perceptron and naive Bayes models call.

It is checked against vectors a JDK 21 wrote; `analysis/jvm/README.md`
says how to regenerate them.

## Offsets

Every offset is a UTF-8 byte offset into one text view, half-open, as a
`u32`: the unit the tokenizer's spans already use (docs/tokenizer.md).
Both ends of a span must be on character boundaries: a layer with a
span that cuts a character is refused when it is added, not found later
by a panic.
UTF-16 and code point offsets, for Java and the wire, are conversions
at the edge, never a second coordinate system inside. Each view has an
`OffsetTable` (`View::offsets`, `Document::offsets`), built the first
time it is asked for and kept: a checkpoint every 256 bytes, stored by
column, that holds the byte, UTF-16 and code point offsets of the first
character boundary there. A conversion is a binary search over the
checkpoints and a forward count over at most one stride of the text; a
view that is all ASCII keeps no checkpoints, since all three units
agree. A UTF-16 offset between the two halves of a surrogate pair, or
any offset past the end of the view, is refused with `Error::Offset`.

`Utf16Offset` and `CodePointOffset` are their own types, so neither can
be passed where a byte offset is expected. `OffsetCursor` converts a
column of offsets in text order (a layer's starts, or its ends) in one
forward pass over the text, and `OffsetTable::column_to_utf16` and
`column_to_code_point` use it to fill a vector the caller provides.

## Views

The source text never changes. A step that changes text (Unicode
normalization, dehyphenation) writes a new `View`, with an `Alignment`
to its parent: the edits, in text order on both sides. Outside the edits
the two texts are the same byte for byte. Mapping an offset either way
is a binary search over the edits. An offset inside an edit widens to
the whole edit (a start to its start, an end to its end), so a mapped
span covers everything the original covered.

A normalizer writes its text through a `ViewBuilder`, which records the
alignment as it goes, so no view exists without one. Each `replace` call
is one edit, never merged with the edit before it, so the caller chooses
the granularity: three characters replaced one call each keep a
one-to-one mapping per character, and a run replaced in one call is one
edit over the whole run. A layer's spans
name the view they are in, and `Document::to_source` maps them back
through any chain of views to exact source bytes.

Inserted or deleted bytes (an edit with one empty side) belong to
neither neighbour: a start offset at them maps to after them and an end
offset to before them. An empty span at them stays empty and lands
before them, at the smaller of the two mapped offsets, so mapping any
span either way never builds a span that ends before it starts. Deleting
the whole text, for example, maps the empty view span at 0 to the empty
source span at 0. An empty span strictly inside an edit widens to the
whole edit like any other span. `Document::offset_from_source` and
`Document::offset_to_source` map one offset through the chain of views
for the side given, so a start at an insertion point goes after every
byte inserted there and an end goes before them.

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
and the layers it indexes. They are the instances its step wrote before
it or read (for a layer added by hand, those its provenance lists, or
the default ones),
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
it. A layer made on another output of its own step (a confidence column
on the tokens written beside it) lists that output apart, under `on`,
since the step did not read it.

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

An aligned output may hang on an earlier output of the same step: a
tokenizer that writes `(Tokens, Confidence<Tokens>)` scores the tokens it
wrote, without reading them. Outputs are added in the order of the
`Writes` tuple, so the base must come before the layer aligned to it,
and the aligned layer is made on the instance the step wrote under its
own key (named or default). The same holds for an index into an earlier
output: tokens written with their sentences point into those sentences.

## The analyzer

`Analyzer::builder()` takes the steps in any order, and the layers the
documents arrive with (`given`), such as structure from the parser
upstream. `build` refuses a pipeline that cannot run, naming why: two
layer types under one id, a layer two steps write, a required input
nothing writes, an aligned output whose anchor the step neither reads
nor writes before it, an aligned output written before its own base, or
a cycle. It then orders the steps by what they read and write into
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

## Golden files

An annotator that ports an OpenNLP component is checked against what
OpenNLP 3.x computes on the JVM. The JVM runs once, from a pinned
OpenNLP commit with fixed models, over fixed input texts, and writes
what each annotator produced as a golden file: typed layers, the same
ones the Rust annotator writes. Rust tests read the files and never
need a JVM. `turbo-goldens` reads and writes the files and compares a
layer on a `Document` with its golden layer.

The input texts are `analysis/goldens/data/inputs/<set>/<id>.txt`, the
file's bytes being the text, and each golden is
`analysis/goldens/data/<annotator>/<model>/<set>/<id>.golden`: one per
annotator (`sentdetect`, `tokenize`, `langdetect`, ...), model (its file
name without `.bin`, or `none`) and input. `turbo_goldens::data_dir()`
is that directory, for tests in any crate.

### The format

A golden file is UTF-8 text with LF line ends and one tab between
fields, ending with a line end. Everything is written in a fixed order,
so making a file again gives the same bytes. The first line is
`golden 1`, with a space: the format version, and a reader refuses any
other. Then one header record per line, a keyword and its fields:

- `generator <name> <version>`: what wrote the file.
- `opennlp <commit> <version>`: the OpenNLP commit, as 40 lowercase hex
  digits, and its Maven version.
- `jdk <java.version> <java.vm.version> <java.vendor>`.
- `unicode <major>.<minor>`: the Unicode version the JDK's character
  tables implement (15.0 for JDK 21).
- `input <set>/<id> <bytes> <units> <sha256>`: the input's length in
  UTF-8 bytes and in UTF-16 units, and its SHA-256 in lowercase hex.
- `model <file name> <sha512>`: zero or more, one per model file the
  annotator loaded, in the order loaded.
- `annotator <class> <method>`: the Java class and method that ran,
  such as `opennlp.tools.sentdetect.SentenceDetectorME` and
  `sentPosDetect`.
- `config <key> <value>`: zero or more, one per option the annotator
  was run with, sorted by key, each key once, so two goldens of one class
  run with different options are told apart. Key and value are `str`
  fields, escaped as below.

The other header fields are written as they are; none holds a tab or a
line end.
One or more layer blocks follow. Each starts with
`layer <id> <anchor> <rows> <name:type>...`, where the anchor is `text`,
`aligned:<layer id>` or `document` and the list names every column with
its type. Then come the rows, one per line, a field per column, and
`end <id>` closes the block. A text anchored layer's rows are in text
order, and an aligned layer has as many rows as the layer it is aligned
to, which comes earlier in the file.

A column is one of three types. A `u32` is decimal, with no sign and no
leading zero (except `0` itself). An `f64` is the 16 lowercase hex
digits of its IEEE 754 bits, a space, and the decimal Java's
`Double.toString` prints, as in `3fe0000000000000 0.5`; the bits are the
value, and the decimal is there for a person reading the file. A `str`
is UTF-8 text in which a tab, a line feed, a carriage return and a
backslash are written `\t`, `\n`, `\r` and `\\`, and every other
character below U+0020 is `\uXXXX`, four lowercase hex digits naming one
UTF-16 unit. Nothing else is escaped. A reader also takes a character
outside the Basic Multilingual Plane written as two `\u` units (a
surrogate pair), and refuses a lone surrogate.

Every `start` and `end` is a UTF-8 byte offset into the input text,
half open, the unit the Rust layers use. The generator computes it from
OpenNLP's UTF-16 span and fails rather than round an offset that falls
inside a character.

The layers and their columns, in order:

| Layer id | Anchor | Columns |
|---|---|---|
| `opennlp:sentences` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:sentences>` | `aligned:opennlp:sentences` | `value:f64` |
| `opennlp:tokens` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:tokens>` | `aligned:opennlp:tokens` | `value:f64` |
| `opennlp:pos` | `aligned:opennlp:tokens` | `tag:str`, `prob:f64` |
| `opennlp:lemmas` | `aligned:opennlp:tokens` | `lemma:str` |
| `opennlp:entities` | `text` | `start:u32`, `end:u32`, `token_start:u32`, `token_end:u32`, `kind:str`, `prob:f64` |
| `opennlp:language` | `document` | `code:str`, `confidence:f64` |
| `turbo:view` | `text` | `start:u32`, `end:u32`, `with:str` |

`opennlp:language` has a row per language the detector ranked, best
first. A `turbo:view` block is what a normalizer produced: a row per
edit, in text order, where `start..end` is the span of the source text
the edit replaces (empty for an insertion) and `with` is the
replacement (empty for a deletion). The JVM writes adjacent one-to-one
replacements as separate rows, one per character.

The ids are the layer types' own (`Layer::ID`, and `Layer::OF`
for a generic layer, printed as `turbo:confidence<opennlp:tokens>`), and
the reader refuses an id no layer type has, a second block for one id,
and a block whose anchor or columns, by name and type, are not its
type's. Every error names the line and what was expected there.

### Comparing a layer with its golden

```rust
use turbo_goldens::{Golden, Tolerance, assert_matches_golden, data_dir};

// tokenize/<model>/<set>/<id>.golden, for one model and one input
let golden = Golden::read(data_dir().join("tokenize").join(model).join(set).join(format!("{id}.golden")))?;
let text = golden.input_text(data_dir())?; // checks the length and SHA-256
let doc = run_the_tokenizer(text);
assert_matches_golden::<Tokens>(&doc, &golden, Tolerance::exact());
assert_matches_golden::<Confidence<Tokens>>(&doc, &golden, Tolerance::abs(1e-9));
```

The default instance of the layer is compared; `matches_golden_key`
and `assert_matches_golden_key` take a `Key` for a named one, and
`matches_golden` returns the difference as a `Mismatch` instead of
panicking. The document's text must be the golden's input, by length
and SHA-256. Spans are compared in source bytes, so a layer on a
normalized view is mapped back through the view's alignment first. Item
by item, the span, then an entity's token range, then the label (a tag
through its tag set, a lemma, a kind, a language code) must be equal,
and a probability must be within the `Tolerance`: `Exact` compares the
bits (so -0.0 and 0.0 differ), and `Abs(d)` takes two finite values at
most `d` apart, or the same infinity, or two NaNs. `Language` holds the
best language, so it is compared with row 0. POS tags without
probabilities do not match a golden that has them.

A normalized `View` is compared by its key (`Key::named("nfkc")`):
the source with the golden's edits applied must be the view's text,
byte for byte, and then the golden's rows must be the view's alignment
edits, one for one and in order, so row i is edit i and a report names
the first row that differs. The JVM writes one row per character for
one-to-one replacements, so a normalizer matches it by making one edit
per character too. Only a view made from the source is compared: a
view made from another view is reported as such, since composing a
chain's edits into edits against the source is not built.

The first difference is reported for a person: the layer, the item, the
spans on both sides with the text they cover (for an aligned layer, the
spans of the tokens or sentences it describes), both labels, and both
values in hex bits and decimal. When one side has more items, the
report gives both counts and the first item past the shorter side:

```
turbo:confidence<opennlp:tokens> item 3: the value differs (by 3.000000026176508e-9, tolerance 1e-9)
  rust:   bytes 15..17 "to", value 3fef5c28f75ee07a (0.980000003)
  golden: bytes 15..17 "to", value 3fef5c28f5c28f5c (0.98)
```

The crate writes golden files too (`Golden::write`), byte for byte as
the generator does: its `Double.toString` follows the JDK 19 and later
definition, which picks the nearest of the shortest decimals and breaks
a tie to an even digit. SHA-256 and SHA-512 are written in the crate
from FIPS 180-4.

## Evaluation

`turbo-eval` (`analysis/eval/`) is the harness every change to the
chunker is to be measured with (#154). Step 1 is in: a library with the
measures, the significance tests, the record and the gate. Reading the
evaluation sets, running a pipeline over them, the named baselines, the
records under `analysis/eval/records` and the `turbo-eval` binary are
later steps.

- `metrics`: nDCG@10 with linear graded gain and a `log2(rank + 1)`
  discount, and recall at 1, 5, 10 and 100, at document level. Hits
  with the same score are ordered by document id, descending, as
  `trec_eval` orders them. Queries without a relevant document are left
  out of the mean, as `trec_eval` leaves them out; a query with relevant
  documents and no ranking scores 0 and counts.
- `spans`: for the top k chunks of a query, as UTF-8 byte ranges of
  their documents, the precision and recall of relevant bytes covered,
  and the fraction of relevant spans that lie wholly inside one chunk.
- `significance`: a paired bootstrap of the mean per query difference
  (10,000 resamples, 95 percent percentile interval) and a paired sign
  flip permutation p value, both drawn from turbo-jvm's `JavaRandom`
  with a fixed seed, so the same records give the same bits.
- `cost`: wall time per stage (analysis, chunking, embedding,
  retrieval), tokens embedded, chunk count and the chunk length
  distribution (min, p50, p90, max) in bytes and tokens.
- `record`: one JSON record per set and pipeline, written with the
  pretty printer turbo-bench uses. Its provenance block (commit,
  `pushed_to`, machine, device, bundle) reuses the core's
  `turbo::record` types. Every field is required, an unknown field is
  an error, and the record carries each query's nDCG@10 so two records
  can be compared query by query.
- `gate`: candidate records against baseline records, one per set. It
  passes when at least two sets gain 2.0 nDCG@10 points or more and no
  set loses more than 0.5, and reports each difference with its
  bootstrap interval and p value beside it.

`cargo test -p turbo-eval` checks the measures against values worked
out by hand in the tests, the tie and empty query rules, the
reproducibility of both tests, the permutation p value where the exact
answer is known, and the record's round trip and its refusal of unknown
and missing fields.

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

## Tests

`cargo test -p turbo-analysis -p turbo-analysis-layers -p turbo-chunker -p turbo-jvm -p turbo-goldens`
runs them on any machine:

- `analysis/core/tests/pipeline.rs` runs a pipeline of toy annotators
  over the shared layers: ordering by data, a normalized view whose
  tokens land on exact source bytes, a batch reaching a batching
  annotator in one call, failures, panics and skips per document, gold
  and predicted layers side by side and never read across, one type
  per id, and a pipeline from configuration.
- `analysis/core/tests/empty_spans.rs` maps empty spans at insertions
  and deletions, at both ends of the text and after the whole text is
  deleted, single offsets through a chain of two views, and every span
  of every alignment four `replace` calls can make over a three-byte
  text, checking that none panics.
- `analysis/chunker/tests/chunker.rs` chunks hand-made layers: a cut
  where the topic shifts instead of where the budget fills, sections
  that are never crossed, and a paragraph break's token counted inside
  a chunk. The chunker's unit tests hold the optimizer against filling
  each chunk to the budget.
- `analysis/core/tests/offsets.rs` converts offsets in both directions
  at every character boundary of text with one to four byte
  characters, refuses offsets inside a surrogate pair and past the end,
  and checks a text of about ten thousand bytes against counts made
  with `encode_utf16` and `chars`, by point and by cursor.
- `analysis/jvm/tests/jvm_vectors.rs` compares `turbo-jvm` with the
  results a JVM wrote, bit for bit.
- `analysis/goldens/tests/goldens.rs` reads a hand written golden file
  (`tests/fixtures/zurich.golden`, over a short input beside it) into
  typed fields, as it does a second with options and a normalized view
  (`tests/fixtures/ligature.golden`), writes each back byte for byte,
  and compares documents built to match them, layer by layer.
  Deliberate differences (a span shifted by a byte, a changed tag, a
  value just outside the tolerance, a missing row, a golden layer with
  the wrong columns, another language code, tags without probabilities,
  a view's text or edits) each fail with a report naming the item, its
  text and both values, and a malformed file of each kind is refused
  with its line. The unit tests hold the escapes, the FIPS 180-4
  test vectors and Java's `Double.toString`.
- A doctest on `Ix` checks that an index into the wrong layer does not
  compile.

## Not built yet

- The annotators themselves: normalizers, sentence detection,
  tokenization, the subwords from turboembed's tokenizer, vectors
  through turboembed, and the boundary signals. The tests build their
  layers by hand or with toy annotators.
- The rest of the evaluation harness: the set readers, the pipeline,
  the baselines, the committed records and the `turbo-eval` binary.
- A gRPC surface, to be shared with the OpenNLP standard.
