# TurboEmbed

TurboEmbed runs text encoder models and the document analysis around
them as native code, with no Python at run time. It is a Rust workspace
of 71 crates that builds four things:

- `libturbo`, a library with a hand-written C API
  (`include/turbo/turbo.h`) that embeds text, reranks passages and
  answers extractive questions on CPUs, NVIDIA GPUs, Intel Arc GPUs,
  Intel NPUs, Apple silicon and Hailo accelerators. Each backend is
  written against the vendor's own low-level interface.
- A document analysis stack: typed annotation layers over exact source
  offsets, normalization, segmentation, structure, Rust ports of the
  Apache OpenNLP components, neural extractors (GLiNER, GLiNER2,
  GLiREL), entity and relation graphs, and a chunker that picks chunk
  boundaries by an exact optimization.
- `turbo-kserve`, a gRPC server that serves bundles over the KServe v2
  Open Inference Protocol and adds analysis, search, sink, model
  catalog, vocabulary and health services.
- Tools that build and verify model bundles, enforce model and dataset
  licences, measure every backend against vendor reference programs,
  and train and evaluate the classic models.

The embedding library, its backends, precision tiers and design are
described in [docs/embeddings.md](docs/embeddings.md).

## Inference library

The C API has three tasks:

| Task | Input | Output | Page |
|---|---|---|---|
| `TURBO_TASK_EMBED` | text or token ids | one pooled vector per row, or one per token range | [embeddings.md](docs/embeddings.md) |
| `TURBO_TASK_RERANK` | query and passage pairs | one relevance logit per pair | [reranking.md](docs/reranking.md) |
| `TURBO_TASK_EXTRACTIVE_QA` | question and context windows | start and end logits, with abstention | [extractive-qa.md](docs/extractive-qa.md) |

A caller creates a runtime, loads a bundle, opens a session with
options (pooling MEAN, CLS or LAST; normalization; truncation; query or
document prompt; precision tier; autotune), writes inputs and reads the
result. `turbo_embed_write_ranges` pools one vector per token range from
a single encoder pass, which is how per-sentence and per-chunk vectors
come from one run over a document. Buffers can be imported from and
exported to native device handles (CUDA pointers, OpenCL `cl_mem`,
Level Zero USM, `MTLBuffer`, dmabuf file descriptors), so vectors can
stay on the device. Every result reports where each stage ran, the
bytes that crossed to the device and back, and the allocations made in
the run.

The tokenizer is exposed through the same header (`encode`,
`encode_pair`, `encode_spans`, `decode`, `count`). WordPiece, Unigram,
byte-level BPE (GPT-2, Llama 3 and Qwen2 pre-tokenizer patterns) and
SentencePiece `.model` files are implemented in the library and checked
against reference token ids ([tokenizer.md](docs/tokenizer.md)).

Model families: BERT and RoBERTa encoders (XLM-R position offsets
included), SaT sentence segmentation encoders with lookahead attention,
and static Model2Vec style models, whose output is checked bit for bit
against Model2Vec in CI ([static.md](docs/static.md)). Bundles can
declare Matryoshka output widths. The Rust API of the core adds SPLADE
sparse terms from a masked language model head, per-token vectors
through a linear projection head, classification heads, and a CPU
DeBERTa-v3 encoder for the GLiNER family ([deberta.md](docs/deberta.md)).

## Hardware backends

| Backend | Cargo feature | Hardware | Code | Page |
|---|---|---|---|---|
| CPU | default | any x86-64 or aarch64 CPU | `core/src/cpu` | [cpu.md](docs/cpu.md) |
| CUDA | `cuda`, `cuda-cublas` | NVIDIA GPUs, Jetson Orin | `core/cuda` | [cuda.md](docs/cuda.md) |
| Level Zero GPU | `levelzero`, `levelzero-onednn` | Intel Arc Xe2 (Arc Pro B70) | `core/levelzero` | [levelzero.md](docs/levelzero.md) |
| Intel NPU | `npu` | Core Ultra NPU, OpenVINO IR compiled on the machine | `core/src/npu` | [npu.md](docs/npu.md) |
| Metal | `metal` | Apple silicon | `core/metal` | [metal.md](docs/metal.md) |
| Hailo | `hailo` | Hailo-10H (Raspberry Pi 5), Hailo-8 | `core/hailo` | [hailo.md](docs/hailo.md) |

Each session picks a precision tier: `MODEL` (the model's own
precision, F32 on most backends), `FASTEST` (F16 on CUDA tensor cores,
Intel XMX engines and Metal, INT8 on Hailo) or `EXACT` (F32). Each tier
is held to a cosine bound against the bundle's reference vectors:
0.9999 for F32, 0.999 for F16 and BF16, 0.93 for INT8
([conformance.md](docs/conformance.md)).

`turbo_runtime_capability` reports a status for every device, task and
precision. A cell is SUPPORTED only when a committed benchmark record
under `benchmarks/records/` backs that device class, precision and
model; any other cell that runs is EXPERIMENTAL, and a cell that cannot
run is UNSUPPORTED with a reason. The library does not fall back to
another device on its own. Records are committed for an RTX 4080 Super
and a Jetson Orin (CUDA), an Arc Pro B70, an Arrow Lake NPU, an Apple
M2, a Hailo-8, a Hailo-10H, and x86-64 and aarch64 CPUs.
`turbo_runtime_select` names the fastest device for a task, with a
one-line reason. GEMM and attention kernel choices are measured per
device and shape and cached ([autotune.md](docs/autotune.md)).

## Bundles and the model zoo

A bundle is a directory holding a manifest with hashes, the weights,
the tokenizer, the model's settings and reference vectors.
`turbo-bundle` (`bundle/`) builds one from a recipe: it fetches the
pinned upstream files, runs the conversion and reference pipeline in
pinned containers with no network, writes the manifest, and verifies
the result with the library's own loader. Its commands are `make`,
`fetch`, `reference`, `seal`, `verify`, `distill`, `catalogue`,
`torch-convert`, and converters and verifiers for DeBERTa, GLiNER,
GLiREL and POS tagger checkpoints ([bundle.md](docs/bundle.md),
[setup/bundles.md](docs/setup/bundles.md)).

Recipes in `bundle/recipes/`: all-MiniLM-L6-v2 (and a static variant),
bge-small, bge-base (and static), bge-large-en-v1.5, bge-m3 (512 and
8192 tokens), bge-reranker-base, ms-marco-MiniLM-L6-v2,
dslim/bert-base-NER, nli-MiniLM2-L6-H768, roberta-base-squad2, SaT 1l-sm
and 3l, and nine potion static models.

`turbo-zoo` (`analysis/zoo`) is an index of models, heads, lexicons,
centroid sets and datasets (`analysis/zoo/index/zoo.json`), each with
its licence, attribution and training data. Its fetcher refuses an
artifact whose licence class has not been accepted with `turbo-zoo
accept` ([zoo.md](docs/zoo.md), [licences.md](docs/licences.md)).

## Document analysis

Analysis runs over a document model in `analysis/core`. Every
annotation is a typed, columnar layer whose offsets are UTF-8 byte
offsets into a view, and every view maps back to the exact source bytes
through an alignment. Offsets convert to UTF-16 units and code points
for callers in other languages ([analysis.md](docs/analysis.md)).

- **Input and text.** WHATWG decoding of 40 encodings with detection
  and mojibake repair (`analysis/encoding`), a Zstandard decoder
  (`analysis/zstd`), Unicode 18.0 data generated from the UCD
  (`analysis/unicode`), and normalizers that write a new aligned view:
  whitespace, quotes, composition, URL placeholders, dehyphenation, plus
  a text quality annotator ([normalizers.md](docs/normalizers.md),
  [text-quality.md](docs/text-quality.md)).
- **Structure.** Sections, headings, paragraphs, lists, tables, code
  and quotes from plain text, Markdown and HTML (`analysis/structure`),
  table and key/value field extraction, and projection of structure
  across normalized views ([tables.md](docs/tables.md),
  [structure-projection.md](docs/structure-projection.md)).
- **Segmentation.** OpenNLP sentence detection, newline and paragraph
  detection, SaT neural sentence boundaries on the library's encoder
  (`analysis/sat`), OpenNLP word tokenizers, UAX #29 word boundaries, a
  MeCab style lattice tokenizer with prebuilt dictionaries, and Chinese
  unigram segmentation ([segmentation.md](docs/segmentation.md)).
- **Language.** OpenNLP language detection and per-span language and
  script detection for mixed-language text
  ([langdetect.md](docs/langdetect.md)).
- **Classic OpenNLP components.** Rust ports that read OpenNLP `.bin`
  models and are checked against the JVM output: POS tagging,
  lemmatization, phrase chunking, constituency and dependency parsing,
  name finding (maxent, CRF, dictionary and pattern), document
  categorization, coreference, stemmers (Porter, Snowball, light),
  stopwords in 11 languages, Hunspell, Morfologik, WordNet, glossaries
  and n-gram language models. See
  [docs/beyond-opennlp/rust-crates.md](docs/beyond-opennlp/rust-crates.md)
  for the full list.
- **Numbers, dates and entities.** A rule grammar for numbers, units,
  money, dates, times, durations and sets; money with ECB rates,
  quantities and temporal expressions; a gazetteer with Natural Earth
  places; PII scanners; Schwartz-Hearst abbreviations; and a token
  pattern matcher in the style of spaCy's Matcher, PhraseMatcher and
  SpanRuler ([grammar.md](docs/grammar.md), [money.md](docs/money.md),
  [quantities.md](docs/quantities.md), [temporal.md](docs/temporal.md),
  [pii.md](docs/pii.md)).
- **Neural extraction.** Transformer token classification NER on every
  encoder backend ([namefind-dl.md](docs/namefind-dl.md)), GLiNER
  zero-shot NER ([gliner.md](docs/gliner.md)), GLiNER2 schema extraction
  in one encoder pass ([schema-extract.md](docs/schema-extract.md)),
  GLiREL and GLiNER2 relations, dependency-path relation patterns,
  predicate-argument events and modality (negation, factuality,
  deontic) ([relation.md](docs/relation.md), [events.md](docs/events.md),
  [modality.md](docs/modality.md)).
- **Documents as graphs.** Entity linking against pluggable knowledge
  bases, consolidation into one entity record per document, entity
  salience, quotation attribution, discourse units with rhetorical
  relations and NLI claim links
  ([entity-graph.md](docs/entity-graph.md),
  [entity-salience.md](docs/entity-salience.md),
  [coref.md](docs/coref.md), [discourse.md](docs/discourse.md)).
- **Classification.** Maxent, SVM, multilabel and zero-shot document
  categorization, sentiment, dense classifiers trained on embeddings,
  and calibration with split conformal prediction
  ([doccat.md](docs/doccat.md), [sentiment.md](docs/sentiment.md),
  [dense-classifier.md](docs/dense-classifier.md),
  [calibration.md](docs/calibration.md)).
- **Embeddings as analysis steps.** Subword pieces with exact spans,
  sentence and chunk vectors from one encoder pass, sparse terms,
  reranking, QA and role-aware routing between loaded models
  (`analysis/embed`, [model-routing.md](docs/model-routing.md)).
- **Keywords and grouping.** YAKE and TextRank keywords
  ([keywords.md](docs/keywords.md)), near-duplicate grouping, version
  links and boilerplate detection (`analysis/group`).

## Chunking, search and vector storage

`turbo-chunker` (`analysis/chunker`) chooses chunk boundaries the way TeX
chooses line breaks: it scores every candidate boundary from signal
layers (structure, topic shift, coreference, discourse) as integer
costs and finds the cheapest segmentation by dynamic programming, so
the same document gives the same chunks on every machine
([chunker-cut-identity.md](docs/chunker-cut-identity.md),
[chunker-text-signals.md](docs/chunker-text-signals.md)). It also holds
the BM25 posting index.

`turbo-vector-index` (`analysis/vector-index`) is an exact in-memory
vector index with SIMD dot and cosine scoring, optional 2 to 4 bit
TurboQuant storage through the `turbovec` crate
([turboquant.md](docs/turboquant.md)), and MaxSim scoring for token
vectors. `turbo-query` analyzes short queries into terms and filters
(temporal, money, quantity, place, entity). `turbo-eval` measures
retrieval quality and gates it, including a SPLATE harness
([splate.md](docs/splate.md)).

## gRPC server

`turbo-kserve` (`server/`) loads bundles from `--model` flags and serves
these services on one port ([grpc.md](docs/grpc.md)):

| Service | Proto | What it does |
|---|---|---|
| `inference.GRPCInferenceService` | `server/proto/open_inference_grpc.proto` | KServe v2 Open Inference Protocol: embed, rerank and extractive QA through `ModelInfer` ([kserve.md](docs/kserve.md)) |
| `turbo.result.AnalysisService` | `server/proto/turbo_analysis.proto` | capabilities, bundle load and unload, `AnalyzeStream` (bidirectional bulk analysis with per-document answers, micro-batching and backpressure), `AnalyzeDocumentProgressive`, document graph and window inspection, chunk projection, and output formats (CoNLL-U, brat, TSV, CSV, Markdown, JSON-LD, WARC) ([grpc-analysis.md](docs/grpc-analysis.md), [grpc-inspect.md](docs/grpc-inspect.md)) |
| `turbo.search.TurboSearchService` | `server/proto/turbo_search.proto` | in-memory indexes fed by a stream, hybrid search over BM25, dense, TurboQuant, SPLADE sparse and token MaxSim providers with score or reciprocal rank fusion, parent expansion and cross-encoder reranking, persist and seal, reindex, aliases and collections |
| `turbo.sink.TurboSinkService` | `server/proto/turbo_sink.proto` | receives analyzed documents; with `--sink-endpoint` the server streams its results to a sink through a bounded queue ([grpc-sink.md](docs/grpc-sink.md)) |
| `turbo.training.TurboCatalogService` | `server/proto/turbo_training.proto` | the model zoo catalog, installed models, licence acceptance and streamed installs ([grpc-catalog.md](docs/grpc-catalog.md)) |
| `turbo.vocabulary.TurboVocabularyService` | `server/proto/turbo_vocabulary.proto` | dictionary import, corpus vocabulary learning and download ([grpc-vocabulary.md](docs/grpc-vocabulary.md)) |
| `grpc.health.v1.Health` | `server/proto/health.proto` | standard gRPC health checks |

Writes that change server state (bundle loading, index persistence,
vocabulary writes) need `--admin`. The server speaks plaintext gRPC with
no authentication, so it belongs behind a trusted network boundary.

`turbo-kserve-demo` (`demo/`) is a small web page that proxies JSON to a
running server for trying embeddings, precision tiers and analysis from
a browser ([demo.md](docs/demo.md)).

## Training and evaluation

`turbo-train`, `turbo-train-components` and `turbo-train-recipes`
(`analysis/train*`) train maxent (GIS, quasi-Newton, perceptron), naive
Bayes, liblinear and CRF models, and reproduce models from permissive
data such as MASC and Universal Dependencies with accuracy gates.
`turbo-bpe-trainer` trains byte-level BPE and other tokenizers
([bpe-trainer.md](docs/bpe-trainer.md)). `turbo-goldens` produces golden
files from OpenNLP on the JVM in a container, and `turbo-fuzz` runs
differential fuzzing against it.

`turbo-bench` (`bench/`) measures the library through its C API against
pinned reference programs (Hugging Face TEI, TensorRT, OpenVINO,
HailoRT, MLX) and writes the records that capability reports cite
([benchmarks.md](docs/benchmarks.md)). The tree holds 171 embedding
records plus range, POS, tagging and throughput records.

## Quick start

Build the library and run the C example on the CPU against the small
test bundle in `testdata/`:

```sh
cargo build --release -p turbo                    # add --features cuda, levelzero, npu, metal or hailo
cc -std=c11 -I include docs/setup/embed.c -L target/release -lturbo \
    -Wl,-rpath,$PWD/target/release -o embed
./embed testdata/tiny-bert-bundle 0 model
```

Build a bundle and serve it over gRPC. A potion static model needs no
container; the transformer recipes run their reference pipeline in
Docker and need the reference image built first
([setup/bundles.md](docs/setup/bundles.md)):

```sh
cargo run --release -p turbo-bundle -- make \
    minishlab/potion-base-8M upstream/potion-base-8M bundles/potion-base-8M
cargo build --release -p turbo-kserve             # add --features cuda for CUDA
target/release/turbo-kserve --listen 127.0.0.1:8081 \
    --model bundle=bundles/potion-base-8M,device=0,sessions=2
```

`scripts/setup/check.sh` reports what a machine is missing for each
backend, and each backend script installs it with `--install`
([setup/README.md](docs/setup/README.md)).

Run the tests the way CI does, with the core in its own target
directory (#358):

```sh
cargo test --workspace --exclude turbo
CARGO_TARGET_DIR=target/core cargo test -p turbo
```

## Releases

A `v<version>` tag builds two archives with SHA-256 sums on a GitHub
release: `linux-x86_64` (CPU and CUDA backends) and `macos-arm64` (Metal
and CPU). The Linux archive carries `libturbo.so`, the `turbo-kserve`
server and the C headers; the macOS archive carries `libturbo.dylib`
and the headers. Other platforms build from source ([release.md](docs/release.md),
[setup/release-archive.md](docs/setup/release-archive.md)).

## Module layout

| Path | Contents |
|---|---|
| `core/` | `libturbo` (crate `turbo`): runtime, bundle loader, tokenizers, sessions, heads, CPU and NPU backends in `core/src`, and the CUDA, Level Zero, Metal and Hailo backends in their own directories |
| `include/turbo/` | the C header and the backend interface |
| `bundle/` | `turbo-bundle`, recipes, and the conversion containers |
| `bench/` | `turbo-bench`, reference and peer programs, and the repository hygiene tests |
| `benchmarks/records/` | committed benchmark records |
| `server/` | `turbo-kserve` and its protos |
| `demo/` | `turbo-kserve-demo` |
| `analysis/core`, `analysis/layers` | the document model, views, alignments and shared layer types |
| `analysis/unicode`, `analysis/encoding`, `analysis/zstd`, `analysis/normalize` | text input and normalization |
| `analysis/structure`, `analysis/sentdetect`, `analysis/sat`, `analysis/tokenize`, `analysis/langdetect`, `analysis/langspan` | structure, segmentation and language |
| `analysis/maxent`, `analysis/sequence`, `analysis/postag`, `analysis/lemmatizer`, `analysis/phrase-chunker`, `analysis/parser`, `analysis/depparse`, `analysis/namefind`, `analysis/doccat`, `analysis/coref` | OpenNLP ports |
| `analysis/stem`, `analysis/stopword`, `analysis/hunspell`, `analysis/morfologik`, `analysis/wordnet`, `analysis/glossary`, `analysis/ngram`, `analysis/abbrev` | lexical resources |
| `analysis/numeric`, `analysis/grammar`, `analysis/geo`, `analysis/pii`, `analysis/matcher` | numbers, dates, places, PII and pattern matching |
| `analysis/embed`, `analysis/namefind-dl`, `analysis/gliner`, `analysis/schema-extract`, `analysis/relation`, `analysis/entity-linker`, `analysis/calibration` | neural annotators, extraction and calibration |
| `analysis/chunker`, `analysis/vector-index`, `analysis/query`, `analysis/group`, `analysis/keywords`, `analysis/eval` | chunking, retrieval and evaluation |
| `analysis/train`, `analysis/train-components`, `analysis/train-recipes`, `analysis/bpe-trainer`, `analysis/formats` | training |
| `analysis/zoo`, `analysis/goldens`, `analysis/goldens-tool`, `analysis/fuzz`, `analysis/jvm` | the model zoo and OpenNLP parity testing |
| `docs/` | documentation; `docs/setup/` has per-machine setup |

Crates with a `-layers` or `-gen` suffix next to these hold layer types
and data generators. [analysis.md](docs/analysis.md) has the crate by
crate table.

## Project rules

The rules every change follows are in [AGENTS.md](AGENTS.md). Among
them: no regex crate (pattern work goes through `turbo-matcher`), no
external NLP dependencies, every Rust file under 500 lines, model
settings in the bundle rather than in code, and no result called working
until it has run on the real device. CI runs formatting, clippy, an
aarch64 check, four test shards, fuzzing, feature builds, parity against
OpenNLP on the JVM and against Model2Vec, and Windows lint and tests.

## Documentation

| Area | Pages |
|---|---|
| Library and backends | [embeddings.md](docs/embeddings.md), [cpu.md](docs/cpu.md), [cuda.md](docs/cuda.md), [levelzero.md](docs/levelzero.md), [npu.md](docs/npu.md), [metal.md](docs/metal.md), [hailo.md](docs/hailo.md), [autotune.md](docs/autotune.md), [conformance.md](docs/conformance.md) |
| Tasks and models | [reranking.md](docs/reranking.md), [extractive-qa.md](docs/extractive-qa.md), [static.md](docs/static.md), [deberta.md](docs/deberta.md), [tokenizer.md](docs/tokenizer.md) |
| Bundles, zoo and benchmarks | [bundle.md](docs/bundle.md), [zoo.md](docs/zoo.md), [licences.md](docs/licences.md), [benchmarks.md](docs/benchmarks.md), [reference-inventory.md](docs/reference-inventory.md) |
| Setup | [setup/README.md](docs/setup/README.md), [setup/bundles.md](docs/setup/bundles.md), [setup/release-archive.md](docs/setup/release-archive.md) |
| Analysis model and text | [analysis.md](docs/analysis.md), [normalizers.md](docs/normalizers.md), [normalization-xray.md](docs/normalization-xray.md), [text-quality.md](docs/text-quality.md), [whitespace.md](docs/whitespace.md), [codepoint-lowercase.md](docs/codepoint-lowercase.md) |
| Structure and segmentation | [tables.md](docs/tables.md), [structure-projection.md](docs/structure-projection.md), [structure-conformance.md](docs/structure-conformance.md), [segmentation.md](docs/segmentation.md), [lattice-prebuilt.md](docs/lattice-prebuilt.md), [unigram-chinese.md](docs/unigram-chinese.md), [langdetect.md](docs/langdetect.md) |
| OpenNLP ports | [beyond-opennlp/rust-crates.md](docs/beyond-opennlp/rust-crates.md), [beyond-opennlp/other-languages.md](docs/beyond-opennlp/other-languages.md), [liblinear.md](docs/liblinear.md), [crf-namefinder.md](docs/crf-namefinder.md), [dependency-parser.md](docs/dependency-parser.md), [feedforward-dependency-parser.md](docs/feedforward-dependency-parser.md), [doccat.md](docs/doccat.md), [sentiment.md](docs/sentiment.md), [coref.md](docs/coref.md) |
| Numbers, dates and PII | [grammar.md](docs/grammar.md), [grammar-units.md](docs/grammar-units.md), [numbers.md](docs/numbers.md), [money.md](docs/money.md), [quantities.md](docs/quantities.md), [temporal.md](docs/temporal.md), [pii.md](docs/pii.md) |
| Neural extraction and graphs | [neural-annotators.md](docs/neural-annotators.md), [namefind-dl.md](docs/namefind-dl.md), [gliner.md](docs/gliner.md), [schema-extract.md](docs/schema-extract.md), [gliner2-relations.md](docs/gliner2-relations.md), [glirel-relations.md](docs/glirel-relations.md), [relation.md](docs/relation.md), [events.md](docs/events.md), [modality.md](docs/modality.md), [entity-graph.md](docs/entity-graph.md), [entity-salience.md](docs/entity-salience.md), [discourse.md](docs/discourse.md) |
| Classification and calibration | [dense-classifier.md](docs/dense-classifier.md), [model-routing.md](docs/model-routing.md), [calibration.md](docs/calibration.md) |
| Chunking and retrieval | [chunker-cut-identity.md](docs/chunker-cut-identity.md), [chunker-text-signals.md](docs/chunker-text-signals.md), [keywords.md](docs/keywords.md), [turboquant.md](docs/turboquant.md), [splate.md](docs/splate.md), [quality-consumers.md](docs/quality-consumers.md) |
| Training and evaluation | [bpe-trainer.md](docs/bpe-trainer.md), [namefinder-evaluation.md](docs/namefinder-evaluation.md), [namefinder-crf-evaluation.md](docs/namefinder-crf-evaluation.md), [lemma-case-evaluation.md](docs/lemma-case-evaluation.md) |
| Server and demo | [grpc.md](docs/grpc.md), [kserve.md](docs/kserve.md), [grpc-analysis.md](docs/grpc-analysis.md), [grpc-inspect.md](docs/grpc-inspect.md), [grpc-sink.md](docs/grpc-sink.md), [grpc-catalog.md](docs/grpc-catalog.md), [grpc-vocabulary.md](docs/grpc-vocabulary.md), [demo.md](docs/demo.md) |
| Releases | [release.md](docs/release.md) |

## License

Apache-2.0; see [LICENSE](LICENSE). The CUDA backend includes a subset
of NVIDIA's CUTLASS headers under `core/cuda/cutlass/` (BSD-3-Clause;
see `core/cuda/cutlass/LICENSE.txt`). Models and datasets fetched by
the tools carry their own licences, recorded in the zoo index.
