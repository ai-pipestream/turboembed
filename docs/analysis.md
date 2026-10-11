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
depends on the add-on that produces it; an add-on's own layers live in
the add-on.

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

A layer is added once and never changed. Before it is taken the
document checks that the layer is not already there, that its spans are
in bounds and on character boundaries, that an aligned layer is as long
as its anchor, and the layer's own invariants. A document clones
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

## Provenance and status

Every layer records where it came from: the annotator and its version,
a hash of its configuration, the model (bundle id, revision and
sha256), where the model ran (backend, device, precision), and the
layers it read. An optional input is listed only when the document had
it.

A layer's status is kept apart from its data. A layer never written has
no status; one written with no items is `Complete` and empty; one its
annotator declined is `Skipped`, with the reason. An annotator declines
a document by returning `Error::Skipped`. A step that needs a skipped
layer is skipped in turn, and says why: `opennlp:language was skipped:
no language evidence`. So "skipped" is never mistaken for "found
nothing".

## Annotators

An `Annotator` declares what it reads and writes as tuples of layer
types, up to six of each. One that reads `(View, Tokens)` and writes
`(PosTags,)` is handed a `&View` and a `&Tokens` and must return a
`PosTags`. An input it uses only when it is there is written
`Option<L>`, and arrives as `Option<&L>`.

`annotate` handles one document. `annotate_batch` handles a whole batch
at once: by default it calls `annotate` on each, and a neural annotator
overrides it to pack every document's input into one device run. An
annotator that returns `Error::Annotator` fails that document alone.

`Bound` binds an annotator to the instances it reads and writes:
the default ones, or named ones (`.read(0, "folded")`).

## The analyzer

`Analyzer::builder()` takes the steps in any order, and the layers the
documents arrive with (`given`), such as structure from the parser
upstream. `build` refuses a pipeline that cannot run, naming why: a
layer two steps write, a required input nothing writes, an aligned
output whose anchor the step does not read, or a cycle. It then orders
the steps by what they read and write into stages.

`run` takes a batch of documents. Each stage runs over the whole batch
before the next starts, so a batching annotator sees every document at
once; steps in one stage compute side by side on threads, then store
their layers. Each document gets its own result, and one document's
failure does not stop the others. `plan` lists the stages.

## The registry

A pipeline can also be assembled from configuration: a list of (name,
spec) lines, where a spec is a model location and options in order. Each
name is resolved through a `Registry<dyn Step>`, whose providers each
have a name, a priority, a cheap availability check and a claim on a
spec. The best available provider that supports the spec wins, and two
that tie at the top priority are an error, not a silent pick. Add-ons
are registered by the program that links them; nothing is found by
scanning. The shape is OpenNLP's `ComponentProvider`.

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
the chunks. Every length is counted in the embedder's own tokens, so a
chunk's size is exact.

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
after each sentence. Signal annotators (structure, topic shifts,
coreference, discourse markers) are summed into it, and the chunker
reads only that.

With structure, no chunk crosses a top-level section. Each section is
its own run, so an edit re-chunks only its own section, and each
section becomes a chunk one level up, the parent of the chunks cut
from it. Every chunk records its level, its parent, its sentences, why
it starts where it does, what it cost the optimizer and its length in
model tokens.

## Tests

`cargo test -p turbo-analysis -p turbo-analysis-layers -p turbo-chunker`
runs them on any machine:

- `analysis/core/tests/pipeline.rs` runs a pipeline of toy annotators
  over the shared layers: ordering by data, a normalized view whose
  tokens land on exact source bytes, a batch reaching a batching
  annotator in one call, failures and skips per document, gold and
  predicted layers side by side, and a pipeline from configuration.
- `analysis/chunker/tests/chunker.rs` chunks hand-made layers: a cut
  where the topic shifts instead of where the budget fills, and
  sections that are never crossed. The chunker's unit tests hold the
  optimizer against filling each chunk to the budget.
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
