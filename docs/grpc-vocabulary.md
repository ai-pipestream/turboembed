# The vocabulary store over gRPC

`turbo.vocabulary.v1.TurboVocabularyService`
(`server/proto/turbo_vocabulary.proto`) is served by `turbo-kserve` on the
same port as the inference and analysis services (docs/grpc.md). It keeps
two kinds of artifact in a directory on the server:

- **dictionaries** a client uploads: glossaries, word lists, XML token
  dictionaries, WordNet lexicons. Each upload is parsed whole by the
  analysis crate that owns its format before it is stored, so a stored
  dictionary is one the analysis readers accept.
- **vocabularies** the server learns from a stream of documents: term and
  document frequencies, BM25 idf, and multi-word n-grams kept by their
  pointwise mutual information.

## Turning it on

| Flag | What it does |
|---|---|
| `--vocabulary-dir PATH` | Keeps the store in `PATH`, made when it does not exist. Without it the store holds nothing: the listings are empty and downloads find nothing. |
| `--admin` | Serves the calls that write to the store: `ImportDictionary`, `LearnVocabulary`, `DeleteVocabularyArtifact`. |

A write is refused before any of its stream is read: `UNIMPLEMENTED`
(`TURBO_E_UNSUPPORTED`) without `--admin`, `FAILED_PRECONDITION`
(`TURBO_E_INVALID_STATE`) without a store. The listings and downloads are
served to every client.

## Calls

| Call | What it does |
|---|---|
| `ListDictionaryFormats` | The formats below, which can be imported, and the store's limits. |
| `ListDictionaries`, `ListVocabularies` | The stored descriptors, in artifact id order. |
| `ImportDictionary` | Client streaming. One `ImportDictionaryHeader` (format, language, name, byte size, SHA-256, whether a delimited file has a header row), then the bytes in `chunk` frames. |
| `LearnVocabulary` | Client streaming. One `LearnVocabularyHeader`, then `LearnVocabularyDocument` frames. |
| `DownloadVocabulary` | Server streaming. The artifact's descriptor, then its bytes in frames of at most 64 KiB. |
| `DeleteVocabularyArtifact` | Removes a dictionary or a vocabulary. |
| `BuildSuggestLexicon` | Builds and stores a memory-mapped `TSUGLEX1` stem-to-surface and prefix completion lexicon (`#2341`). |
| `SpellcheckQuery` | Context-aware token-bigram Viterbi spellcheck preserving query syntax and UTF-8 / UTF-16 / code-point spans (`#2341`). |
| `ImportCorpusStatistics` | Aggregates shard document-frequency deltas into a deterministic `TCIDF001` BM25 IDF snapshot (`#2346`). |
| `TrainVoronoiCentroids` | Trains spherical k-means Voronoi routing centroids into a `TCENT001` snapshot (`#2346`). |

Every descriptor carries the artifact's id, byte size and the lowercase
hex SHA-256 of its bytes. A download's bytes always match them: the store
checks the file against its descriptor before sending it, and refuses
with `INTERNAL` a file changed behind its back.

## Dictionary formats

| Format id | Read by | Entries counted |
|---|---|---|
| `glossary-csv` | `turbo-glossary` `DelimitedReader`, comma separated | glossary entries |
| `glossary-tbx` | `turbo-glossary` `TbxReader`, the header's language selected (`en` also selects `en-US`) | glossary entries |
| `glossary-mapped` | `turbo-glossary` `MappedGlossaryMatcher` | glossary entries |
| `plain-word-list` | one word or phrase a line, UTF-8 | non-blank lines |
| `headword-definition-tsv` | a headword, a tab and a definition a line, UTF-8 | headwords |
| `opennlp-dictionary-xml` | `turbo-maxent` `Dictionary` | dictionary entries |
| `pos-tag-dictionary-xml` | `turbo-sequence` `TagDictionary` | words |
| `lemmatizer-dictionary` | `turbo-lemmatizer` `DictionaryLemmatizer` | word and tag pairs |
| `wordnet-lmf-xml` | `turbo-wordnet` `WnLmfReader` | lexical entries |
| `morfologik-fsa` | listed, not importable: a dictionary and its `.info` file are two files | |
| `hunspell-aff-dic` | listed, not importable: an `.aff` and a `.dic` file | |

A glossary read from text (`glossary-csv`, `glossary-tbx`) is also
compiled with `MappedGlossaryWriter` and the image is stored beside the
upload, so a pipeline can map it without compiling.

An upload is refused with `INVALID_ARGUMENT` when its byte count or
SHA-256 is not what the header says, when the format's reader refuses it,
or when it has no entries; a header over `max_dictionary_upload_bytes`
(256 MiB) is refused with `OUT_OF_RANGE` (`TURBO_E_CAPACITY`) before any
byte is read. A refused upload stores nothing. The artifact id is derived
from the format, the language, the header-row setting and the bytes'
digest, so uploading the same file again answers with the stored
descriptor, the first upload's name included.

## Learning a vocabulary

Each document is split into UAX #29 words (`turbo-tokenize`), lowercased
when the header asks. Words are n-gram neighbours while only whitespace
lies between them, so no n-gram crosses punctuation. The header sets:

| Field | Meaning |
|---|---|
| `min_term_frequency` | A term or n-gram seen fewer times is left out (0 counts as 1). |
| `max_terms` | The most single terms, and separately the most n-grams, kept: the most frequent, ties in byte order. 0 keeps all. |
| `filter_stopwords` | Leaves out the language's bundled stopwords (`turbo-stopword`) as terms, and n-grams that start or end with one. The language's primary subtag picks the list; a language with no list is refused. |
| `include_ngrams`, `min_ngram_size`, `max_ngram_size` | Counts n-grams of these sizes, 2 up to `max_ngram_size` in the limits (5). |
| `min_pmi` | An n-gram's PMI must reach this many bits: `log2(P(w1 .. wn) / (P(w1) * .. * P(wn)))`, with `P(g)` the n-gram's count over the n-gram positions of its size and `P(w)` a word's count over all words. |
| `dictionary_artifact_id` | A stored dictionary whose headwords are kept whatever their counts, and marked as seeds. |

At most `max_learn_bytes` (1 GiB) of document text is read. The
vocabulary is tab separated UTF-8 text:

```
# turbo-vocabulary 1
# language=en	documents=3	tokens=32	lowercase=true	filter_stopwords=true	min_term_frequency=2	...
term	kind	tf	df	idf	pmi	seed
office	term	5	3	0.133531		0
new york	ngram	3	2	0.470004	3.245112	0
```

`kind` is `term` or `ngram`, `idf` is BM25's with each document one unit,
`pmi` is empty for a term and for a seed n-gram the corpus never has, and
`seed` is 1 for a seed dictionary's headword. Terms come first, then
n-grams, each most frequent first with ties in byte order. The artifact
id is derived from the bytes, so the same documents in the same order
with the same header are one artifact.

## Binding into an analysis pipeline (issue #2013)

A stored dictionary or learned vocabulary is bound into a running
analysis pipeline without a restart through `VocabularyArtifactRef`
(`artifact_id` and lowercase hex `sha256`, in
`server/proto/turbo_analysis.proto`) inside `VocabularyBinding`:

- `LoadBundleRequest.vocabulary` (field 13) binds a bundle's default
  `stopwords` and `glossary` artifacts when the bundle is loaded and
  advertises the active binding on `BundleCapabilities.vocabulary`
  (field 10).
- `AnalyzeDocument.vocabulary` (field 8) on `AnalyzeStream` and
  `AnalyzePipelineRequest.vocabulary` (field 17) on
  `AnalyzeDocumentProgressive` bind `stopwords` and `glossary` for one
  document or request, overriding whichever slots the request sets over
  any bundle default. On `AnalyzeStream`, setting `output` to
  `ANALYZE_OUTPUT_DOCUMENT_GRAPH` (4) runs the document graph pipeline
  with the resolved binding and answers with
  `AnalyzeStreamResponse.document_graph` (`DocumentGraphResponse`).

Each reference resolves against the `--vocabulary-dir` store and is
verified against its 64-character lowercase hex `sha256` before the
bytes are parsed, then cached by digest so repeated requests share one
compiled annotator:

- `stopwords` accepts a `plain-word-list` dictionary from
  `ImportDictionary` or a `# turbo-vocabulary 1` artifact from
  `LearnVocabulary` (each `term` row's headword, NFKC-normalized and
  lowercased), driving `turbo:stopwords` (`turbo-stopword:filter`) and
  `turbo:keywords` (`turbo-keywords:yake`).
- `glossary` accepts a `glossary-csv`, `glossary-tbx`, or
  `glossary-mapped` dictionary from `ImportDictionary`, or a
  `# turbo-vocabulary 1` artifact from `LearnVocabulary`, driving
  `turbo:glossary-terms` (`turbo-glossary:annotator`, appended after
  tokenization when bound on the default pipeline).

Every step that runs with a bound artifact records
`ModelRef { id: artifact_id, revision: format_id, sha256 }` in its
layer provenance (`LayerInfo.model`, `TypedDocumentResult` layer
provenance, and `ProcessingReceipt` step receipts) and folds the
artifact identity into `config_hash`. A wrong `sha256`, an unknown or
malformed `artifact_id`, an incompatible dictionary format, or a server
started without `--vocabulary-dir` is refused with `INVALID_ARGUMENT`
(`TURBO_E_INVALID_ARGUMENT`, 256) naming the reference and never
quoting dictionary terms or filesystem paths.
