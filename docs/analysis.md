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
| `turbo-chunker` | `analysis/chunker/` | The chunker, and term statistics and BM25 per chunk, section and document |
| `turbo-structure` | `analysis/structure/` | Document structure (`turbo:structure`) from plain text and Markdown |
| `turbo-analysis-embed` | `analysis/embed/` | The embedding model as analysis steps: the subwords of a text view from the bundle's tokenizer, and a vector per sentence and per chunk |
| `turbo-numeric` | `analysis/numeric/` | The number scanner, which writes `turbo:numbers` ([numbers.md](numbers.md)); the money extractor, which writes `turbo:money`, the ECB rate table and the conversion step, which writes the `converted` instance of `turbo:money` ([money.md](money.md)); the quantity extractor, which writes `turbo:quantities` ([quantities.md](quantities.md)) |
| `turbo-jvm` | `analysis/jvm/` | JVM behaviour the OpenNLP ports must reproduce bit for bit |
| `turbo-goldens` | `analysis/goldens/` | Golden layer files from the JVM: the reader, the writer, the comparison, loading golden layers as test inputs, SHA-256 and SHA-512 |
| `turbo-sequence` | `analysis/sequence/` | OpenNLP's beam search, BIO and BILOU codecs and feature generation framework (generators, `StringPattern` and the XML descriptors), with the context generators and validators of the part-of-speech tagger, chunker and name finder, the part-of-speech tag dictionary (`tags.tagdict`) and the Penn Treebank and Universal Dependencies tag mapping. Its crate documentation lists the OpenNLP behaviour it reproduces that is surprising, and the one it does not yet (QN models in a beam search); `analysis/sequence/README.md` describes its JVM golden files |
| `turbo-namefind` | `analysis/namefind/` | OpenNLP's `DictionaryNameFinder`, `RegexNameFinder` and the patterns of `RegexNameFinderFactory`, which write `opennlp:entities` from `opennlp:tokens`, and the merge of several entity layers by `dropOverlappingSpans`. User patterns in Java syntax are translated to the `regex` crate or refused with a typed error; the crate documentation lists what is refused and the OpenNLP behaviour that is surprising, and `analysis/namefind/README.md` describes its JVM golden files |

They need nothing beyond the standard library and each other, except
`turbo-analysis-embed`, which runs the embedding model and so depends on
the core crate, `turbo`, and `turbo-namefind`, which matches its patterns
with the `regex` crate. A shared
layer lives in `turbo-analysis-layers`, so a consumer of a layer never
depends on the add-on that produces it; an add-on's own layers live in a
small schema crate beside it, for the same reason.

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
hold. Vectors (`Vectors<L>`) are f32: they are computed in Rust and
never compared with OpenNLP. The chunker's costs (`Boundaries.cost`,
`Chunks.cost`) are integers, so every backend picks the same cuts.

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
| `Numbers` | `turbo:numbers` | text | `turbo-analysis-layers` |
| `Money` | `turbo:money` (and the instance `converted`) | text | `turbo-analysis-layers` |
| `Quantities` | `turbo:quantities` | text | `turbo-analysis-layers` |
| `Vectors<L>` | `opennlp:embeddings<L>` | `L` | `turbo-analysis-layers` |
| `Confidence<L>` | `turbo:confidence<L>` | `L` | `turbo-analysis-layers` |
| `Utf16Spans<L>` | `turbo:utf16<L>` | `L` | `turbo-analysis-layers` |
| `Language` | `opennlp:language` | the document | `turbo-analysis-layers` |
| `Vocabulary` | `turbo:vocabulary` | the document | `turbo-chunker` |
| `Terms` | `turbo:terms` | tokens | `turbo-chunker` |
| `TermStats<L>` | `turbo:term-stats<L>` | `L` | `turbo-chunker` |

`Chunks` are retrieval chunks. The standard's `opennlp:chunks` are
phrase chunks (noun and verb groups), a different layer, which a phrase
chunker will write under that id.

## Subwords

`SubwordAnnotator` in `turbo-analysis-embed` writes `opennlp:subwords`:
it reads a text view (the source, or a named view through `Bound::read`)
and, when the document has them, the word tokens, and runs the bundle's
own tokenizer (`Tokenizer::encode_spans`, docs/tokenizer.md) over the
view. `SubwordAnnotator::open` loads the bundle and checks its
tokenizer against the reference ids as the core does; `SubwordsProvider`
offers it to a registry as `subwords`, the spec's location being the
bundle directory. The bundle's id and revision and the SHA-256 of its
manifest are the layer's `model` and its provenance's.

- The view is encoded as one row, with no truncation, so the layer
  covers the whole view however long it is; a text past the model's
  `max_seq` is split into chunks later, from these pieces.
- The row has no special tokens from the template and no prompt prefix,
  since neither is in the text. `SubwordAnnotator::overhead(role)` is
  what they add to every row the model sees (the template's special
  tokens and the prefix's tokens, counted on the prefix alone), for the
  chunker's `Budget::overhead`. A byte-level BPE or Unigram prefix can
  merge with the text's first piece, so a row can come out one token
  shorter than the pieces and the overhead.
- Each piece's span is the tokenizer's, a UTF-8 byte span of the view. A
  byte-level piece holding part of a character spans the whole
  character, so two pieces can share a span, and every span is on
  character boundaries. A special token written in the text (`[SEP]`
  typed by the author) is a piece of the text like any other.

With word tokens in the same view and in text order, every piece gets
the token it belongs to (`align`), in one pass over both span lists:

- a piece inside a token belongs to it;
- a piece over two or more tokens (a byte-level `'s` or `.)` across the
  word tokenizer's split) belongs to the first of them;
- a piece in no token (whitespace or punctuation the word tokenizer left
  out) belongs to the token before it, or to the first token when none
  comes before it.

So the pieces of one token are a run, which `Subwords::of` finds by
binary search. Tokens in another view, or out of order, fail the
document. A document with pieces but no tokens gets no token column.

## Vectors

`VectorAnnotator` in `turbo-analysis-embed` writes a vector per sentence
(`opennlp:embeddings<opennlp:sentences>`) and per chunk
(`opennlp:embeddings<turbo:chunks>`). It embeds windows of the subword
pieces once each and pools every item's vector from the encoder's final
states over its pieces, through the core's late pooling
(`turbo_embed_write_ranges` in include/turbo/turbo.h). It drives the core through its
C interface, as any caller does, behind a small safe type (`Encoder`):
a runtime, the model on the device asked for (the CPU, a backend by
name, a device index, or the one `turbo_runtime_select` picks), and one
session at the precision asked for.

Three steps run it, since an aligned layer hangs on a layer the step
reads: `SentenceVectors` reads the subwords and the sentences,
`ChunkVectors` the subwords and the chunks, and `SentenceChunkVectors`
all three and writes both from the same windows. They can share one
loaded model (`Arc<VectorAnnotator>`). `VectorsProvider` offers them to
a registry as `vectors`, the spec's location being the bundle directory
and its options `layers` (`sentences`, `chunks`, `both`), `device`,
`precision` and `prompt`.

The subwords must be the same bundle's pieces (the layer's model is
compared) and every layer must be in the subwords' view; a document
where either fails fails alone.

### Rows and windows

A row is the template's leading special tokens, the prompt's prefix
(the document role by default), the pieces of one window, then the
template's trailing special tokens. The overhead around the pieces is
`SubwordAnnotator::overhead` for the same role, the chunker's budget
overhead, so a chunk the chunker kept within a budget of at most
`max_seq` fits one window. The prefix is the prefix's own tokens, as
the overhead counts them; a byte-level BPE or Unigram prefix that the
tokenizer would merge with the text's first piece is not merged here.
A window holds at most `width` pieces: `max_seq` less the overhead. The
special tokens and the prefix are never pooled; an item is pooled over
its pieces alone.

Every window runs `width` pieces from its first piece, or to the end of
the text. The window policy, for one document:

1. An item covers the pieces whose span overlaps its own, found by
   binary search over the pieces' starts and ends. A piece that an
   item's edge cuts is covered, as the chunker counts it on both sides.
   A piece of no bytes at offset `x` is covered by an item `[start, end)`
   when `start <= x < end`: at the item's first byte or inside it, not at
   its end. An empty item covers no piece, whatever sits at its offset.
2. An item that covers no piece (an empty span, or whitespace a
   WordPiece tokenizer drops) is not placed.
3. The windows are laid out by the finest layer the step has: the
   sentences for `SentenceVectors` and `SentenceChunkVectors`, the chunks
   for `ChunkVectors`. Only the items of that layer that fit in `width`
   pieces take part. Sorted by their first piece, then their last, each
   goes in the current window when that window holds all of it;
   otherwise a new window starts at its first piece and becomes the
   current window.
4. Every other item is placed once those windows exist: first the items
   of the laying layer longer than `width`, then the items of the other
   layer (the chunks, beside sentences), each group in the same order.
   When an existing window holds the item whole, it is pooled there, in
   the window with the lowest first piece that does. Otherwise it is
   covered by the windows: from its first piece `p`, the window holding
   `p` that reaches furthest gives the part from `p` to that window's
   end, or to the item's end if that comes first, and the next part
   starts there. Only where no window holds `p` is a new window added,
   starting at `p`. Each of the item's pieces is in exactly one part.

So the windows depend on the laying layer and never on the longer items
beside it. A sentence is pooled from the same window, with the same
bits, by `SentenceVectors` and by `SentenceChunkVectors`. A chunk over
sentences that cover all of its pieces adds no window, however long it
is; a chunk over text that no sentence covers (a gap between two
sentences longer than what the windows reach) adds windows there. A
sentence that would cross a window's end starts the next window, so
windows overlap where that happens.

A chunk's vector depends on which layer laid out the windows. The same
chunk can get one vector from `ChunkVectors`, where the chunks lay out
the windows, and another from `SentenceChunkVectors`, where the
sentences do: the windows differ, so the context each piece is encoded
in differs. The configuration hash says which layer laid out the
windows, so the two are told apart in provenance.

The windows of every document in a batch go through the same runs: up
to the session's `max_batch` rows and `max_ranges` ranges a run (16 per
row unless set). A window whose ranges do not all fit in a run goes
again, with the rest, in the next.

### What a pooled vector is

A vector is the mean of the encoder's final states over the item's own
pieces. The template's special tokens and the prompt's prefix are in
every row, so they shape the context each piece is encoded in, but they
are never pooled. So a sentence's vector here is not the vector the
core's text interface (`turbo_embed_write_text`) gives for the same
span: that pools over every token of its row, the special tokens and
the prefix included, and encodes the span with only its own text in
context.

The prompt of the vector step (`VectorOptions::prompt`, the `prompt`
option) must be the one the chunker's budget was counted with
(`SubwordAnnotator::overhead` for that role, as `Budget::overhead`).
The chunker takes the overhead as a number and the chunks layer does not
record the prompt, so the vector step cannot check this: a pipeline that
counts the budget for one role and embeds with another gets rows longer
or shorter than the chunker planned for, and a chunk the chunker kept
within `max_seq` may no longer fit one window.

### Items longer than a window, and items with no pieces

An item that no window holds whole cannot be pooled from one pass of
the encoder. Its vector is the mean of its parts' vectors weighted by
their pieces: the mean of all its pieces' final states, each piece
encoded in the context of its own part's window. The model cannot read
more than `max_seq` tokens in one pass, so there is no single pass to
match for an item longer than a window, and refusing such an item would
fail every document with a chunk at a coarse level of the hierarchy (a
section over many windows). Attention never crosses from one part's
window to the next.

An item that covers no piece has no token to pool. Its row is the zero
vector, so the layer stays aligned to its items, and `Vectors::empty`
lists it (in order, each once; the layer's check holds every listed row
to zeros). Cosine similarity is undefined for a zero vector: a reader
skips the items `Vectors::is_empty` names rather than comparing them or
indexing them.

### Normalization and provenance

The core is asked for the plain means, and the vectors are normalized
as the bundle says on the host, with the arithmetic of the core's CPU
backend (the norm summed in f64, at least 1e-12), so a split item is
normalized after its parts are combined. A bundle pooled by its first
or last token is refused: late pooling is the mean, which is not the
vector such a model was trained to give.

Every vector layer's provenance has the bundle's id, revision and
manifest hash as its model, and as its execution the backend, the
device's name, and the precision as the dtype the session computes in
and the tier asked for, such as `f32 (model)`. The configuration hash
covers the row length, the prompt, and the layer that lays out the
windows: the sentences for `SentenceVectors` and `SentenceChunkVectors`,
the chunks for `ChunkVectors`.

The annotator checks what it is given, since `VectorAnnotator::embed`
takes its input from any caller: an id for every piece, the pieces'
starts and ends in text order, and no span that ends before it starts;
a document that fails any of these fails alone. A backend that pools no
ranges (one without `embed_write_ranges`) is refused when the session
is made or at the first write, with an error naming the backend and the
range write the vector steps need. A result of another shape than the
run was written for, or a read of fewer bytes, is an error, never a
shorter set of vectors.

## Structure

`Structure` (`turbo:structure`) is a document's blocks as a tree:
sections, headings, paragraphs, lists and list items, tables with their
rows and cells, code blocks and block quotes (`Block` also has captions,
footnotes and page headers and footers, for extractors of HTML and PDF
layout to write). Each block has a span, a kind, a parent and a depth:
a heading's or a section's level, a list's, a list item's or a quote's
nesting (1 for the outermost), and 0 for the rest. The blocks are in
text order with each parent before its children; the layer's check
refuses a parent at or after its child, a block outside its parent's
span, a block that starts before the one before it, and two siblings
that overlap. A shared column of kinds rather than a layer per kind keeps
the nesting (a code block inside a list item inside a quote) in one
place for the chunker to read.

`StructureAnnotator` in `turbo-structure` reads a `View` and writes the
layer; `detect` does the same outside a pipeline, and
`StructureProvider` offers it to a registry as `structure`, with no model
location and one option, `syntax`: `markdown` (the default) or `plain`.
`ParagraphDetector` in `turbo-sentdetect` writes the same layer with
paragraphs alone (docs/segmentation.md); a pipeline runs one of the two,
since a layer has one writer.
It is written by hand, with no Markdown library: one pass over the
lines, keeping the open quotes and list items and at most one open
paragraph, code block or table inside them, as CommonMark's reference
parser does. A line ends at `\n`, `\r\n` or a lone `\r`, as for
`ParagraphDetector`. Every span starts at its block's first character
that is not whitespace (the marker, for a quote, a list item, a heading
or a code fence) and ends after its last one, so line breaks, trailing
blank lines and the indentation of a first line are outside it. A line
is blank when it holds only whitespace (`char::is_whitespace`, so a line
of U+3000 or U+00A0 is blank too). A byte order mark at the start of the
view is in no block. A thematic break (`***`, `---`) ends the block before
it and is no block itself.

Sections come from the headings at the top level: a section runs from
its heading to the end of the last top-level block before the next
heading of the same or a higher level, and holds the heading, the blocks
after it and the sections of lower levels. A document whose first
heading is level 2 has level 2 sections at the top. Blocks before the
first heading are in no section, and a heading inside a quote or a list
item opens none. The chunker cuts each top-level section on its own.

With `markdown`, the blocks are CommonMark's: ATX and setext headings,
paragraphs, block quotes with lazy continuation lines, bullet and ordered
lists (an item continues while a line is indented to its content, an
ordered item interrupts a paragraph only when it starts at 1, an empty
item never does, and a list holds the items of one bullet character or
one delimiter), and fenced and indented code. Tabs stop every four
columns. A list item's indentation is measured from where its parent's
content starts on each line, as CommonMark does, so `> > 1.  one` and
`>>     two` continue one item. The hand-labelled fixtures in
`analysis/structure/tests/fixtures.rs` pin these rules down, including
the cases of the CommonMark specification that are easy to get wrong;
the crate is not tested against the specification's examples or against
another Markdown parser, since it takes no dependency on one.
Front matter is skipped: when the view starts (after a byte order mark,
if any) with a line `---` and a later line is `---` or `...`, the lines
up to and including that one are in no block, as static site generators
read YAML metadata. CommonMark has no front matter and reads such a
start as a thematic break and a setext heading. Without the closing
line, or anywhere but the start, `---` is a thematic break as usual.
What it does not do: HTML blocks and link reference definitions are
paragraphs, footnote definitions are not recognized, and a list is not
marked tight or loose.

GFM pipe tables are tables: a delimiter row (`|---|:-:|`, with at least
one `|`) under a line with as many cells turns that line into the
header, splitting it from the paragraph above when there is one. The
rows run to a blank line or a line that starts another block, and every
row (the header included, the delimiter row not) is a `TableRow` with a
`TableCell` per cell, the cell's span trimmed, an empty cell an empty
span where the cell starts. A leading and a trailing `|` are not cells.
A `|` escaped with a backslash or inside a backtick code span does not
split a cell. That differs from GFM, where a pipe inside a code span
splits the cell unless it is escaped: a pipe inside code (`a | b`, a
shell pipeline) is almost always the code's own, and splitting there
gives a row more cells than its header. A body row keeps the cells it
has; it is not padded or cut to the header's width.

With `plain`, a text is read as a letter or a contract: paragraphs
between blank lines, the same list items (and `•` as a bullet; an
ordered item's number has at most three digits, so `2024. A great year`
is text), quotes marked with `>`, and headings of two kinds. A line of
its own underlined with three or more `=` (level 1) or `-` (level 2):
an email's signature line `-- ` is no underline. And a line at the top
level in capitals (at least three letters, an upper case one among
them, no lower case one, so `OK` is not one), at most 80 characters
long, not ending with `.`, `,`, `;` or `:` (level 1), when it is a
paragraph of its own or the first line of a paragraph whose second line
has lower case letters (`INTRODUCTION` over `This report covers the
year.`), from which it is split off. Such headings in capitals with
nothing after them but more of the same, like a name and a country
under a letter's closing, are paragraphs, so they open no section. An
underline under a paragraph of several lines ends the paragraph and is
no heading. There are no `#` headings, no code, no tables and no front
matter, so an indented line is text.
Tables of aligned columns in plain text are not detected.

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
closes it plus a size penalty: `Budget::size_weight` times the squared
distance from the target size over the squared target, rounded down to
an integer. The optimizer finds the cut of least total cost. No chunk
is longer than the budget's maximum, unless one sentence alone is: that
sentence is then a chunk of its own, marked in `Chunks::over_budget`,
for a clause splitter to cut further; its `cut` still says why it starts
where it does (a section, say). Time is the number of sentences times the most
that fit in one chunk.

Every cost is an integer, so the same layers give the same cuts on
every machine and backend, with no float summed in a different order to
tip a choice. Two cuts tie only when they cost exactly the same; then
the one whose first boundary comes earliest wins, then its second, and
so on. The table runs from the end of each run back, so each entry
keeps the earliest end of equal least cost. A chunk's size penalty is
exact, never capped: a size is at most 2^33 tokens and the weight under
2^32, so a penalty is under 2^98 and fits an i128, and the optimizer
sums chunk costs in i128. Two different cuts never look equal for a cap
or an overflow, and a chunk of more than one sentence is never let past
the maximum to escape a huge penalty. A total could saturate only past
2^28 chunks each billions of tokens from the target. The `Chunks` layer
stores each chunk's cost saturated to the i64 range. A unit test checks the optimizer
against trying every cut of up to eleven sentences, with negative and
zero costs, a target of zero and several weights and overheads.

`segment` also takes a chunk term: a cost of a whole chunk, added beside
its size penalty and clamped to plus or minus 2^60 units, since the
caller supplies it. It is how a signal over whole
chunks, such as how tightly a chunk's sentences cluster, enters the
optimizer; the chunker passes `no_term`.

### Boundary costs

What a boundary costs is the `Boundaries` layer (`turbo:boundaries`),
aligned to the sentences: entry i is the gap after sentence i, the last
one the end of the text. Each entry is an `i32` cost and a `Cut`
reason. Costs are in `Boundaries::UNIT`s (1000); a plain sentence
boundary costs one unit, a stronger break less, and a signal that
argues for a cut may write a negative cost. A signal computed in
floating point is rounded to units once, when it is written.

The chunker reads only the default instance, and a layer has one
writer. With one boundary signal, that signal writes the default
instance itself, as `StructureBoundaries` does. With several (structure,
drift, topic shifts, coreference, discourse markers), each should write
a named instance of this same shape, and a step that weighs them with
integer weights and sums them should write the default instance,
computing in i64 and saturating to the i32 range so no weight or sum
wraps around. A new signal is one more instance, never a new column. A
term that belongs to a whole chunk rather than to a gap (how
tightly its sentences cluster) is not a boundary cost; it is the
optimizer's chunk term.

`StructureBoundaries` writes the structural term from the sentences and
`turbo:structure`. A block holds the sentences it has a byte of, found
with two binary searches; it starts at the gap before the first of them
and ends at the gap after the last, so a block edge inside a sentence
moves to the nearest gap. A section is the exception: it holds the
sentences that start in it, the same rule (one shared function) the
chunker splits its runs by, so a section start inside a sentence is the
gap after that sentence, and every chunk marked `Section` starts a run.
Each gap takes its strongest break:

| Break at the gap | Reason | Default cost |
| --- | --- | --- |
| A section starts or ends | `Section` | 0 |
| A heading starts | `Heading` | 100 |
| A paragraph, list, table, code block, quote, caption, footnote or page header or footer starts or ends | `Paragraph` | 400 |
| A list item or table row starts or ends | `Item` | 600 |
| None of these | `Sentence` | 1000 |

A gap where a heading ends costs 4000 more, so a heading stays with what
it heads. With no structure every gap is a sentence boundary. It writes
the default instance, as the only signal.

### Hard limits

A chunk boundary never falls inside a sentence: boundaries are only
between sentences. Nor does it fall inside a table row, a code block or
a quote, unless that unit alone is over the budget's maximum. These
limits depend on the budget, so they are not costs: the chunker reads
the structure itself. A unit holds the sentences it has a byte of. A
gap is held when any unit that holds sentences on both sides of it fits
the budget, and the sentences joined by held gaps are one unit to the
optimizer. When units are around a gap and none fits, the gap may be
cut, and the chunk that starts there is marked `Cut::Inside` with the
smallest such unit's kind. Units nest: a code block that fits inside a
quote that does not is held whole while the quote is cut around it. No
boundary cost, however low, overrides a limit.

Two units whose bytes do not overlap can still share a sentence, when
an edge falls inside it, and a chain of such units can join more
sentences than the budget holds though each unit fits. That run must be
cut somewhere, so every gap in it is freed and a chunk that starts at
one is marked `Cut::Overlap` with the kind of the smallest unit that
held it. A section's edge is always a cut: when a sentence that
straddles a section's start belongs to the section before (by its
start) and a unit in the new section holds it, the unit is cut at the
section's edge and the chunk there says `Section`.

### Sections

With structure, no chunk crosses a top-level section. The chunker sorts
the top-level sections by start, so sections listed out of text order
still make runs of their own. Each section, and
each run of sentences outside every section, is its own run, so an edit
re-chunks only its own section, and each run becomes a chunk one level
up, the parent of the chunks cut from it. Every chunk records its level,
its parent, its sentences, why it starts where it does, what it cost the
optimizer (an integer, in units), its length in model tokens and
whether it is over the budget. The chunks layer refuses a parent that is
not at a level above its child or does not hold the child's span.

Not measured yet: the chunker against the stage 0 baselines on the
evaluation harness, with significance, which is still being built.

## Term statistics and BM25

`turbo_chunker::terms` counts terms per retrieval unit for lexical
search, from the word tokens alone: no model and no vectors, so a BM25
baseline can be measured before any vector exists.

`TermIndexer` reads a text view and the tokens and writes the document's
`Vocabulary` and a term for each token (`Terms`, aligned to the tokens).
A term is the token's text in the view read, byte for byte: nothing is
lowercased here. To match without case, read a case folded view
(`NormalizeStep::new(CaseFold::full())` bound to write `folded`, and the
indexer bound with `.read(0, "folded")`); the tokens can stay on the
source, since a token's span is mapped through the folded view's
alignment to its parent. A view further down a chain than that is
refused. The view's edits must not cross a token edge: with one edit
turning "don't" into "do not" over the tokens "do" and "n't", each
token would map to the whole replacement, so a document with a token
edge strictly inside an edit is refused. Case folding replaces one
character per edit and never crosses one.

A token is a term when it holds at least one character
`char::is_alphanumeric` accepts (an alphabetic character, or a number
of category Nd, Nl or No). A token of punctuation or symbols alone (a
period) is not, and counts toward no length. This is close to Lucene's
`StandardTokenizer`, which drops punctuation, but not the same: that
tokenizer splits by UAX #29 word boundaries, keeps emoji as tokens
(no terms here) and makes no token of a lone No character such as "½"
(a term here), and its splits differ from the word tokenizer's. The vocabulary is in byte order, each term once, so a
term's id depends only on which terms the document holds, and a lookup
is a binary search.

`TermCounter<L>` reads the tokens, their terms, the vocabulary and the
units of one level, and writes `TermStats<L>`, aligned to the units:

- `TermCounter<Chunks>`: a row per chunk, at every level of the chunk
  hierarchy;
- `TermCounter<Structure>`: a row per structure block; the sections are
  the blocks of kind `Section`, and a nested section's terms count in it
  and in every section around it;
- `TermCounter<View>`: one row for the whole document.

A token is in the unit its start is in. Each row is sparse: the unit's
term ids in ascending order with a count for each, by column, with
offsets per unit. The layer also holds each unit's length in terms and,
per term, how many of the level's units hold it (`df`). That df is
within one document; it is not the corpus document frequency BM25's idf
is defined over.

`Query` scores units with BM25 in Lucene's form (`BM25Similarity`,
`Bm25::LUCENE`: k1 = 1.2, b = 0.75):

```text
score = sum over query terms t of idf(t) * tf / (tf + k1 * (1 - b + b * length / mean_length))
idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
```

The classic `(k1 + 1)` factor is left out, as Lucene has since version
8; it scales every score alike and leaves the ranking unchanged. N, the
mean length and n(t) are the collection's, and the caller supplies them
through `Collection`: from an index over the corpus for corpus search.
`LocalCollection` gives one document's own units as the collection (all
of a level's units, or those a filter keeps, such as the sections or
the chunks of one level), for ranking a document's chunks against each
other. As in Lucene, a unit with no terms is not counted in N or the
mean length. Lengths here are exact, where Lucene stores a length in
one byte, exactly only for short units, so Lucene's scores for long
units can differ slightly. Query terms are matched byte for byte and
must be normalized as the document's terms were; a term given twice
counts twice. `Query::rank` lists the units holding a query term, best
first, equal scores in unit order.

The layers live in `turbo-chunker` beside the code that makes them while
only it reads them; they move to a schema crate of their own once
another crate does.

### Comparing two runs

`turbo_chunker::compare` checks whether two runs of the chunker over one
document cut it in the same places: two backends, or two precision tiers
on one backend. It takes each run's `Chunks`, where the model behind its
costs ran (the `Execution`: backend, device and precision tier, which the
caller takes from the layer the costs came from, since the chunker runs
no model) and, when the caller has them, the sentences, boundary costs
and budget the run was cut with. Runs whose chunks are in different
views are refused, as is a run with two rows at one place or with level
0 rows that overlap under one parent, and a run whose costs do not fit
it: sentences in another view or out of text order, a boundary cost
missing, or a chunk that does not start and end where a sentence does.

A cut is a chunk's start, as an exact byte offset, at its level, under
its parent (named by the parent's span). Cuts both runs make are
counted; the rest are reported in stretches between two shared cuts, in
order of level, parent and offset. A chunk's cost depends only on its
own sentences and the boundary that closes it, so the optimizer's
choices on either side of a shared cut are independent, and each
stretch is one choice: a moved cut is a stretch with a cut on each side,
an added cut one with a cut on one side only. Every cut in a stretch
carries why its chunk starts there, whether it is over the budget,
what the chunk cost its run (from the layer) and what each run's
boundary costs charge at that offset.

The layer records no runner-up, so each stretch's margin is computed:
how much more a run's own costs charge for the other run's cuts there
than for its own, in units. Each chunk from the shared cut on is priced
with the optimizer's own cost of one chunk (the boundary that closes it,
none at the end of its parent, plus its integer size penalty), so the
margin is exact and is what the run's whole cut would cost more had it
cut the stretch as the other did: at least the gap between its best and
second best choice, and zero for a tie. A test checks this against
changes of one to three gaps of the optimizer's cut on generated runs
of one and two sections. As a rule of thumb, a margin smaller than what
the precision tiers move a boundary cost by points at precision as the
cause.

A margin is computed only at level 0 (the level the optimizer cuts),
only after a shared cut, only where the other run's cuts are sentence
starts of this run's, and only where both choices end at the same byte.
Children of parents whose spans differ are never in one stretch, so
they share no cut and have no margin. It adds no chunk term, since the
layer records none, and it prices as if allowed a cut the run's own
limits would forbid (inside a unit it keeps whole) and a chunk over the
run's maximum. Each choice is priced with the pricing run's budget but
sized by its own run's token counts, which agree when the runs share a
tokenizer; runs with different tokenizers mix two counts. There is no
command for it: no record format holds chunks yet.

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
- `error <class> <message>`: only for a run where the annotator threw
  on the JVM, after the config records and the last line of the file.
  The class is the exception's, such as
  `java.lang.IllegalArgumentException`, and the message is a `str`
  field, what `getMessage()` returned (empty for null). OpenNLP's
  `SimpleTokenizer` with `keepNewLines`, for one, throws on a text that
  starts with a line break, where the Rust port tokenizes it.

The other header fields are written as they are; none holds a tab or a
line end.
A file with an error record has no layer blocks, and the reader refuses
any line after that record. Otherwise one or more layer blocks follow. Each starts with
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
OpenNLP's UTF-16 span. OpenNLP's `SimpleTokenizer` and `TokenizerME`
classify UTF-16 units, so they can split a character outside the Basic
Multilingual Plane, a surrogate pair, between two tokens, and a byte
offset cannot fall between its halves. For such a row the generator
widens the byte span to the enclosing character (the start rounds down,
the end rounds up), so the two halves of one character are two rows
over the same bytes, and it writes the aligned layer
`turbo:utf16<opennlp:tokens>` (or `turbo:utf16<opennlp:sentences>`)
whenever at least one row cuts a character. That layer has, for every
row of the layer it is aligned to, the JVM's exact span in UTF-16 units
(`start16`, `end16`, half open) and `cut`, 1 when the row starts or ends
inside a surrogate pair and 0 otherwise. Overlapping rows on the text
are allowed as they are for nested entities (a row may not start before
the row before it), so the widened rows need no rule of their own. The
reader checks that a UTF-16 span does not end before it starts or past
the input's length in UTF-16 units.

The layers and their columns, in order:

| Layer id | Anchor | Columns |
|---|---|---|
| `opennlp:sentences` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:sentences>` | `aligned:opennlp:sentences` | `value:f64` |
| `turbo:utf16<opennlp:sentences>` | `aligned:opennlp:sentences` | `start16:u32`, `end16:u32`, `cut:u32` |
| `opennlp:tokens` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:tokens>` | `aligned:opennlp:tokens` | `value:f64` |
| `turbo:utf16<opennlp:tokens>` | `aligned:opennlp:tokens` | `start16:u32`, `end16:u32`, `cut:u32` |
| `opennlp:pos` | `aligned:opennlp:tokens` | `tag:str`, `prob:f64` |
| `opennlp:lemmas` | `aligned:opennlp:tokens` | `lemma:str` |
| `opennlp:entities` | `text` | `start:u32`, `end:u32`, `token_start:u32`, `token_end:u32`, `kind:str`, `prob:f64` |
| `opennlp:language` | `document` | `code:str`, `confidence:f64` |
| `turbo:numbers` | `text` | `start:u32`, `end:u32`, `numeral_end:u32`, `negative:u32`, `coefficient:str`, `scale:u32`, `plain:str` |
| `turbo:money` | `text` | `start:u32`, `end:u32`, `coefficient:str`, `negative:u32`, `scale:u32`, `plain:str`, `currency:str` |
| `turbo:quantities` | `text` | `start:u32`, `end:u32`, `coefficient:str`, `negative:u32`, `scale:u32`, `plain:str`, `unit:str` |
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

The layer types the reader knows are a parameter. `Golden::read`,
`Golden::parse` and `Golden::parse_bytes` check blocks against
`known_shapes()`, the types of `turbo-analysis` and
`turbo-analysis-layers`; `read_with`, `parse_with` and
`parse_bytes_with` take the set. A crate that owns a layer type with a
golden shape (a tokenizer crate's `turbo:word-types`, a word type per
token in one `kind:str` column) implements `GoldenCompare` for it and
reads its goldens with `known_shapes()` plus `shape_of::<ItsLayer>()`.

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
and SHA-256. When the golden has an error record, the comparison
reports it as `Mismatch::JvmError` with the class and the message
instead of comparing anything, so a test for an input the JVM could not
handle asserts that error on purpose and then checks the Rust output
itself. Spans are compared in source bytes, so a layer on a
normalized view is mapped back through the view's alignment first. Item
by item, the span, then an entity's token range, then the label (a tag
through its tag set, a lemma, a kind, a language code) must be equal,
and a probability must be within the `Tolerance`: `Exact` compares the
bits (so -0.0 and 0.0 differ), and `Abs(d)` takes two finite values at
most `d` apart, or the same infinity, or two NaNs. `Language` holds
every language the detector ranked, best first, and is compared row by
row, so the ranking must match. POS tags without
probabilities do not match a golden that has them.

`Utf16Spans<L>` (`turbo:utf16<L>`) is compared exactly, row by row: the
UTF-16 span, then the cut flag, the Rust layer's `bool` against the
golden's 0 or 1 (a golden with any other value is reported as one that
cannot be applied). Where the JVM split a character the Rust tokenizer
keeps whole, the byte comparison of the tokens shows that as an
expected deviation the test asserts on purpose: the golden has two rows
over the character's bytes where Rust has one token, so the report is
a count mismatch when the character ends the text (`Smile 😀`: 2 items
in Rust, 3 in the golden) and otherwise the first token after it, which
the golden still has over the character. The UTF-16 layer keeps the
JVM's own answer, the two halves with `cut` 1.

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

### Golden layers as test inputs

A port is often written before the annotators in front of it: a POS
tagger before the Rust tokenizer, a name finder before the tagger. To
test it alone, `turbo-goldens` builds a document from the layers the JVM
produced, and the port runs on the JVM's own input layers:

```rust
use turbo_goldens::{GOLDEN_KEY, Golden, Tolerance, data_dir, golden_key, load_document, matches_golden_key};

let golden = Golden::read(path)?;
let text = golden.input_text(data_dir())?;
let mut doc = load_document(&golden, &text)?; // checks the length and SHA-256
// The tagger reads the JVM's tokens and writes the default instance of the tags.
let tagger = Bound::new(Tagger::new(model)).read(1, GOLDEN_KEY);
Analyzer::builder().given(golden_key::<Tokens>()).step(tagger).build()?.run(slice::from_mut(&mut doc));
assert_eq!(matches_golden_key(&doc, &Key::<PosTags>::DEFAULT, &golden, Tolerance::abs(1e-9)), Ok(()));
```

`load_document` adds every layer block of the golden, in file order,
each under the key name `golden` (`GOLDEN_KEY`; `golden_key::<L>()` is
`Key::<L>::named("golden")`). The JVM's instance of a layer is thus a
named instance, and the default key stays free for what the Rust
annotator writes, so the two are never confused and can be compared.
`load_layer` adds one layer under a key the caller picks; loading the
tokens under two names gives two separate instances.

A layer aligned to another (tags, lemmas and confidences) is loaded
onto the instance of its anchor with the same name, and the entities
onto the tokens with the same name, since their `token_start` and
`token_end` columns index those tokens. That instance must be loaded
first; otherwise the load fails with `LoadError::MissingAnchor`. Spans
are source bytes, values are the f64 values as the file holds them, and
the labels of a tag or kind column become a tag set of the distinct
labels in the order they first appear. Loaded tokens get their
sentence column from the sentences loaded under the same name: each
token's sentence is the one whose span contains it, and a token inside
no sentence is refused (`LoadError::OutsideSentences`), since the JVM
makes its tokens inside its sentences. The format writes sentences
before tokens, so `load_document` finds them; without sentences under
that name the column is `None`. `Language` holds every row of the
golden, best first.

A block is checked against its layer type's shape before it is loaded,
the same check the comparison makes, and the document then checks the
layer as it checks any other (spans in bounds and on character
boundaries, an aligned layer as long as its anchor). Each refusal is a
`LoadError`: the text is not the golden's input, the golden has no such
block, the block has the wrong anchor or columns, the anchor is not
loaded, an entity's token range is outside the tokens, a block of a type
nothing loads, or the document refused the layer.

A loaded layer's provenance names the annotator `golden:<class>`, after
the Java class in the header, with the OpenNLP version as its version
and the layers it is made on as its reads. Its configuration hash is
taken from the SHA-256 of the annotator, the OpenNLP commit, each model's
file name and SHA-512 and each option, so goldens made with different
models or options have different hashes. It records no model:
`ModelRef` holds one model by SHA-256, and a golden names any number by
SHA-512.

A `turbo:view` block is not loaded. A view golden is only compared with
the view a Rust normalizer makes, so `load_document` on one gives the
source text alone, for the normalizer to run on.

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

The generator writes these layers, in this order within a golden:

| Annotator | Layers |
|---|---|
| `sentdetect` (`SentenceDetectorME.sentPosDetect`) | `opennlp:sentences`, `turbo:utf16<opennlp:sentences>` when a sentence cuts a character, `turbo:confidence<opennlp:sentences>` |
| `tokenize` (`TokenizerME.tokenizePos`) | `opennlp:tokens`, `turbo:utf16<opennlp:tokens>` when a token cuts a character, `turbo:confidence<opennlp:tokens>` |
| `langdetect` (`LanguageDetectorME.predictLanguages`) | `opennlp:language` |
| `normalize` (each normalizer) | `turbo:view` |
| `numbers` (`NumberScan.parse`, per notation) | `turbo:numbers` |
| `money` (`CursorMoneyExtractor.extract` in `MoneyAnnotator`, per notation) | `turbo:money` |
| `money-convert` (`EcbFxRates.convert` in `MoneyConversionAnnotator`, per notation) | `turbo:money` |

`numbers/<notation>/num/<id>.golden` (`latin-us`, `latin-eu`) holds
`turbo:numbers`, anchored on the text, with one row per number mention:
`start:u32` and `end:u32` (UTF-8 byte offsets, from an optional `-` to the
end of a `k`, `m`, `b`, `bn` or `thousand` to `trillion` scale marker),
`numeral_end:u32` (the end of the digits, before the scale marker),
`negative:u32` (1 when the mention starts with `-`), `coefficient:str` (the
unscaled value without sign), `scale:u32` (digits after the decimal point)
and `plain:str` (`toPlainString()`, for a human reading the file). The value
is `(-1)^negative * coefficient / 10^scale`. The `annotator` line names
`opennlp.tools.extraction.NumberScan` and `parse`, and a `config notation`
record gives `LATIN_US` or `LATIN_EU`. The scan is the mention loop of
`CursorQuantityExtractor` of the `numeric` module of `opennlp-addons`
(pinned in `analysis/goldens-tool/src/recipe.rs`) without the money
exclusion and without units; there is no model and no dump.

`money/<notation>/money/<id>.golden` and
`money-convert/<notation>/money/<id>.golden` both hold
`turbo:money`, one row per mention, with the columns `start`,
`end` (UTF-8 byte offsets, from the `-`, symbol or first digit to the end
of the mention), `coefficient`, `negative`, `scale`, `plain` (as for
`turbo:numbers`) and `currency` (the ISO 4217 code). Both have `config
notation`; the conversion goldens also have `config target` (`EUR`),
`config as_of` (`2026-03-15`) and `config snapshot`
(`rates/ecb-snapshot.csv`, relative to `analysis/goldens/data`). A mention
without a rate has no row in a conversion golden. The generator reads
the snapshot from the `rates` directory beside the inputs; `turbo-goldens
generate` mounts it at `/rates`. See [money.md](money.md#golden-files).

Other crates write goldens of their own layer types in the same format,
which the generator does not make: a tokenizer crate's
`turbo:word-types`, one `kind:str` per token, aligned to
`opennlp:tokens`, is one.

When the OpenNLP annotator throws on an input, the golden holds the
header records, the `config` records and then, as its last line,
`error <class> <message>`: the exception's fully qualified class and
what `getMessage()` returned (empty for null), escaped as a `str`
field. It has no layer blocks, and that input has no decision point
dump. Only the call into the annotator is caught this way; any other
exception (an input or model that cannot be read, one of the
generator's own checks) stops the generator. None of the three
annotators throws on the checked in inputs.

`TokenizerME` decides at every UTF-16 unit inside a whitespace token, so
it can split a character outside the Basic Multilingual Plane between
two tokens (`'😀.` gives `'`, the high surrogate, the low surrogate and
`.`). A byte offset cannot fall between the two halves, so the row of a
span that starts or ends inside a surrogate pair holds the whole
character (the start rounds down to the character's first byte, the
end up to the byte after it), and the two halves are two rows over the
same bytes. Right after the layer, the generator then writes
`turbo:utf16<opennlp:tokens>` (or `turbo:utf16<opennlp:sentences>`),
aligned to it, with one row per row: `start16:u32` and `end16:u32`, the
JVM's exact half open span in UTF-16 units, and `cut:u32`, 1 for a row
that cut a character and 0 otherwise. The layer is written only when at
least one row cuts a character, so every other golden is as it was.
`analysis/goldens/data/inputs/en/emoji-split.txt` is such an input for
the tokenizer.

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
  `input` lines, one `model` line per model loaded, the same
  `annotator` class and method, and the same `config` records (none of
  the three annotators with dumps is run with options now).
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
and can hold such halves. The tokenizer dumps of `en/emoji`,
`en/emoji-split` and `multi/mixed` have them. The language detector
dumps have none, because its first normalizer replaces every character
outside the Basic Multilingual Plane with a space before the n-grams
are cut; a reader that
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
golden of these three annotators has a dump beside it, except one with
an error record, which has none, that every dump has a golden, and that
each dump's header parses and names the input and model of its path. The
checked in dumps were made by running the generator directly under the
JDK named in their headers, as the goldens were; the container run is to
be proven on a machine with Docker.

## Tests

`cargo test -p turbo-analysis -p turbo-analysis-layers -p turbo-chunker -p turbo-structure -p turbo-analysis-embed -p turbo-jvm -p turbo-goldens`
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
  each chunk to the budget and against trying every cut, check the
  integer size penalty, the chunk term, that of equal cuts the earliest
  wins and that a huge penalty never lets a chunk past the maximum, and
  read the gaps a structure makes: each gap's strongest break, the
  units by their sentences, block edges inside sentences, which gaps
  units hold, and sections listed out of order.
- `analysis/chunker/tests/limits.rs` holds each hard limit over
  hand-made layers: a table row, a code block and a quote that fit are
  never cut even where the costs say to, each one over the budget is cut
  and the chunk says why, a unit that fits inside one that does not is
  held, units that share sentences and chain over the budget are cut,
  a section's edge is a cut even inside a unit, and a sentence over the
  budget stands alone and keeps its reason, at a section start too. It
  refuses chunks outside their parents or not at a level above them. It
  checks that every chunk is the exact bytes of its
  sentences in text of one to four byte characters and, through a
  normalized view, in the source, and the costs `StructureBoundaries`
  writes.
- `analysis/chunker/tests/terms.rs` runs the term steps on hand-made
  tokens, structure and chunks, with the terms read from a case folded
  view: the vocabulary, a row per chunk, block and document with its
  lengths and df, BM25 scores against a plain textbook BM25 written in
  the test for four settings of k1 and b, Lucene's idf, an idf from a
  caller's corpus statistics against the document's own, ranking ties,
  full case folding that changes bytes, a token a unit cuts, units in
  another view and a view too far from the tokens. Two scores are
  worked by hand from Lucene 9's formula in the test's comments. It
  also holds a query term given twice, units with no terms left out of
  the local N and mean, a mean length of zero, a view whose edit crosses
  a token edge (refused) beside one whose edits do not, and the
  counter's refusals: tokens out of order and a layer of several units
  with no spans. The unit tests hold
  the interning, the sparse rows and the layer's checks.
- `analysis/structure/tests/fixtures.rs` holds hand-labelled documents,
  each one's whole structure written out block by block with the text
  each span covers: Markdown with nested sections, setext and ATX
  headings in one hierarchy, fenced code holding heading, list and table
  lines, unclosed fences, indented code, nested and mixed lists with
  tabs and lazy lines, what may interrupt a paragraph, items that start
  blank, tables with escaped pipes and pipes in code, quotes, CR LF and
  lone CR breaks, Unicode text and whitespace, empty and whitespace-only
  documents, front matter, and plain text: headings in capitals over
  their paragraphs, a letter's closing and signature, the `-- ` line and
  a year that is no list item.
- `analysis/structure/tests/exhaustive.rs` runs every document of up to
  three lines from 32 tricky lines, in both syntaxes, with each of the
  three line breaks and with and without a leading byte order mark, and
  holds each layer to the document's checks, to trimmed spans and to
  giving the same layer twice; it also runs ten thousand nested quotes,
  a thousand nested list items, twenty thousand nested list items under
  two hundred thousand blank lines (which must parse in under ten
  seconds; visiting every item on every blank line takes over a minute), a
  table row of 400 unclosed backtick runs and a long document.
- `analysis/structure/tests/pipeline.rs` runs the annotator ahead of the
  chunker, whose chunks then stay inside the sections it found, and
  builds it through the provider.
- `analysis/embed/tests/subwords.rs` runs the subwords annotator on the
  small BERT's WordPiece bundle, a byte-level BPE bundle and BGE-M3's
  Unigram, over the tokenizer test texts, and checks that the document
  takes every layer and that the ids are the tokenizer's. The WordPiece
  pieces are compared with upstream `tokenizers`, id for id and span for
  span. It also covers pieces of one character sharing its span, a text
  many times `max_seq` covered to its end, the overhead, pieces given
  their word tokens (a BPE piece over two tokens, punctuation in no
  token), tokens that cannot be aligned, and the provider. The crate's
  unit tests hold `align` on hand-made spans.
- `analysis/embed/tests/vectors.rs` runs the vector steps on the small
  BERT on the CPU. The row is the tokenizer's own row for each prompt
  role, and its overhead the subwords annotator's; a static model's has
  none. A range over a whole row is the row's mean-pooled vector bit
  for bit, and normalizing it on the host is the core's normalization
  bit for bit. Every sentence and chunk vector (sentences alone, chunks
  alone, both) is held to the same range run alone in a session of its
  own, the parts of a split item combined by hand: in one window,
  across several windows with runs smaller than a window's ranges, for
  a sentence that crosses a window's end, for chunks longer than a
  window, for a chunk over text no sentence covers, and for three
  documents in one batch against each alone. Sentence vectors are the
  same bits with or without the chunks beside them, and a hierarchy of
  chunks short and long adds no window to the sentences'. Items with
  no pieces get the zero vector and are listed in the layer's `empty`,
  and empty layers stay empty. It also checks provenance (the layer
  that lays out the windows included), the provider, subwords of
  another model and sentences in another view, malformed input to
  `embed`, a model pooled by its first token, and rows with no room for
  text. `window.rs`'s unit tests hold the policy on hand-made spans,
  pieces of no bytes included; `encoder.rs`'s hold the result's shape
  and the refusal of a backend that pools no ranges. One test runs the steps on the device
  `TURBO_TEST_DEVICE` names (a backend name or a device index), at the
  precision `TURBO_TEST_PRECISION` names, and holds the vectors to the
  CPU's within the tolerance the core gives that dtype; without the
  variable it says it was skipped. On a CUDA machine:
  `TURBO_TEST_DEVICE=cuda cargo test -p turbo-analysis-embed --features turbo/cuda --test vectors on_the_device`.
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
- `analysis/goldens/tests/load.rs` loads the same fixture as test
  inputs, runs a table lookup tagger on the golden tokens and compares
  its tags with the golden's POS layer (and finds a changed entry). It
  also checks that every loaded layer matches its own golden, that
  golden tokens get the sentence they are in, that two
  names give two instances, and that a missing anchor, a text that is
  not the input, a token straddling two sentences, an entity token
  range past the tokens and a block with
  the wrong columns are each refused.
- `analysis/goldens/tests/jvm_error.rs` reads, writes and compares a
  golden with an error record (`tests/fixtures/newline-first.golden`):
  the message's escapes, a layer block or any other line after the
  record refused with its line, layers beside it refused by the writer,
  and `Mismatch::JvmError` from a comparison.
- `analysis/goldens/tests/utf16_spans.rs` reads and writes a golden
  whose tokens cut a surrogate pair (`tests/fixtures/smile.golden`),
  checks the UTF-16 layer's shape and spans, shows one Rust token over
  the character as a count mismatch against the golden's two rows, and
  compares `Utf16Spans` for tokens and sentences, a span and a cut flag
  that differ each reported with the token's text.
- `analysis/goldens/tests/shapes_from_other_crates.rs` defines a layer
  type in the test (`turbo:word-types`), reads a golden holding it with
  `read_with` (which `read` refuses), and compares a document with it.
- A doctest on `Ix` checks that an index into the wrong layer does not
  compile.

## Not built yet

- The annotators themselves: normalizers, sentence detection,
  tokenization, and the boundary signals besides structure, with the
  step that weighs and sums them.
- A comparison of the sentence vectors with OpenNLP's
  `SentenceVectorsDL` on golden inputs, which needs the JVM container,
  and a measurement of documents per second through the vector steps.
  The tests build their layers by hand or with toy annotators.
- A parity check of the subwords against the JVM's WordPiece encoder on
  golden inputs; the WordPiece pieces are compared with upstream
  `tokenizers` for now.
- The evaluation harness that every change to the chunker is measured
  with, and the chunker's records on it. It is also where the BM25 only
  baseline is to be reported, from the term statistics above.
- A gRPC surface, to be shared with the OpenNLP standard.
