# turbo-postag

OpenNLP 3.x's maxent part-of-speech tagger, `POSTaggerME`, ported to Rust on
the analysis API: `PosModel` opens a `POSModel` file, `PosTagger` tags the
tokens of each sentence by beam search and is the annotator that reads
`opennlp:tokens` and writes `opennlp:pos` (a tag and its probability per
token, aligned to the tokens, with the tag set in the layer). The beam search,
the feature generators, the tag dictionary and the Penn Treebank and Universal
Dependencies tag mapping are `turbo-sequence`'s. The crate documentation
(`src/lib.rs`) describes the modules and where OpenNLP's behaviour is
surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork.

## Classical models and UPOS tagger replacements

The published OpenNLP UD 1.3 taggers are OpenNLP 2.5 artifacts, which are
not fetched, pinned or run (#1509, #1646). The accuracy test on the UD
English EWT test split that read the English one and the JVM goldens under
`analysis/goldens/data/postag` (plus the UD POS entries under `beam`,
`tag-dictionary` and `tag-format`) are deleted (#1646). The fixture model
with a tag dictionary, `pos-fixture.bin`, is trained for this project on
permissive data and checked in as one of the golden generator's fixtures,
in `analysis/goldens/data/fixtures` (that directory's README says where it
comes from).

UPOS and Penn POS tagging are replaced by our own tagger heads on a
shared encoder:

| Language / Tagset | Replacement |
|---|---|
| English (UPOS + XPOS on UD English EWT) | Tagger head on UD English EWT (#1523, user-downloaded CC BY-SA 4.0) |
| English (Penn POS on MASC) | Tagger head on MASC Penn POS (#1649, CC BY 3.0 US, shippable) |
| Spanish, Catalan, Finnish, Italian (UPOS) | UPOS tagger heads on AnCora, FTB and MarkIT (#1522, CC BY 4.0, shippable) |
| German, Portuguese and other CC BY-SA UD treebanks (UPOS) | Recipes under #1651 (user-downloaded CC BY-SA 4.0) |

## BiLSTM-CRF on the CPU backend

`neural::BilstmPosTagger` runs an `ONLP-BLPT` model (char BiLSTM, word
BiLSTM of one or two layers, optional pretrained vectors and adapter,
optional CRF) and is an annotator from `opennlp:tokens` to `opennlp:pos`. It
has two numerics (`neural::Numerics`):

- `Packed`, the default (`BilstmPosTagger::new`). The weights are packed once
  into `turbo::cpu::Matrix`, the CPU backend's f32 GEMM, and every LSTM runs
  on it: the input projections of all the tokens of a call in one product,
  the recurrent product over the sentences still running at each step
  (longest first), the char LSTM once per distinct word with a
  per-character projection table, and the output layer in one product.
  Each row of a product is computed alone, so a sentence's scores do not
  depend on what it is batched with. Word representations are cached (up to
  100,000 words). The annotator batches every sentence of a document in one
  call. The CRF is decoded in f64 by max-plus Viterbi, and the marginals
  (the probabilities) by forward-backward in log space. Where plain f64
  cannot hold a sentence's scores (a bound on its paths' scores above 1e6,
  as with CRF weights near 1e300), that sentence is decoded by the exact
  scorer instead; a probability that is not finite is an error. A weight
  past f32's range does not fail either: if it cannot be packed, `new`
  makes a `JvmParity` tagger (`numerics()` says so, and
  `with_numerics(.., Numerics::Packed)` returns the error), and if it
  overflows a call's f32 scores, that call's emissions come from the f64
  reference.
- `JvmParity` (`BilstmPosTagger::jvm_parity`), for the golden tests and the
  trainer's evaluation only: the f64 forward pass in the JVM's order and the
  JVM's exact-sum CRF scorer, which give the JVM's emissions to within
  rounding.

The two numerics have different configuration hashes (`config_hash`, FNV-1a
over the numerics and the model's SHA-256), so their outputs are never
cached as each other's. `tests/bilstm_numerics.rs` checks both rules
through the public API, and `tests/bilstm_provenance.rs` checks that each
numerics' tags carry its own hash through an analyzer run, the document
graph export, its bytes and the import, and as the key of a pipeline cache.

`tests/neural_goldens/` runs both on every golden of
`analysis/goldens/data/postag-neural/` (six BiLSTM runs). The parity
numerics match the JVM as before (42,924 of 43,560 emissions bit for bit,
the rest within 8.9e-16). The packed numerics
give the JVM's tag on all 1,980 tokens, with emissions within 7.6e-7 and
probabilities within 1.7e-7 of the JVM's (the test's tolerance is 1e-5; tags
must be equal).

### Accuracy and throughput on UD English EWT

No published BiLSTM-CRF model exists, so `tests/bilstm_ewt.rs` trains one on
the EWT train split (trainer defaults, a CRF, five epochs, the dev split for
model selection), then tags the test split with both numerics. The treebank
is CC BY-SA 4.0 and neither it nor the model is in the repository; the test
is ignored unless the splits are named. Sentences are read as syntactic
words (a multiword token's range line skipped and its parts kept, empty
nodes dropped) with their UPOS tags.

```
for s in train dev test; do curl -sSfL -o /tmp/ud/en_ewt-ud-$s.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-$s.conllu; done
TURBO_UD_EWT_TRAIN=/tmp/ud/en_ewt-ud-train.conllu TURBO_UD_EWT_DEV=/tmp/ud/en_ewt-ud-dev.conllu \
TURBO_UD_EWT_TEST=/tmp/ud/en_ewt-ud-test.conllu TURBO_BILSTM_MODEL=/tmp/ud/bilstm-ewt.bin \
TURBO_BILSTM_REPS=3 TURBO_BILSTM_BATCH=32 \
  taskset -c 6 cargo test --release -p turbo-postag --test bilstm_ewt -- --ignored --nocapture
```

With UD English EWT r2.15 (test split SHA-256 as above), training took
1,224 s on one core and wrote an `ONLP-BLPT-2` model of 10,204,385 bytes,
17 tags, SHA-256
`819328e7b1863ac5247e013547cadbfbf74d81ce09f1a04a1cd2030762adfad2`. On the
test split's 2,077 sentences and 25,094 words:

| Numerics | Correct | Accuracy |
|---|---|---|
| `Packed` (default) | 23,823 | 94.9350% |
| `JvmParity` | 23,823 | 94.9350% |

The two give the same tag on all 25,094 words.

Throughput, one core of an AMD Ryzen 9 9950X3D (pinned with `taskset`),
words per second over the test split, the median of six passes of each
(three rounds of packed, parity, parity, packed). A cold pass loads a fresh
model and tagger, so the representation cache starts empty; a warm pass
reuses one tagger. "Emissions alone" is the BiLSTM without the CRF, one
sentence per call.

| Sentences per call | | Packed | JvmParity | Ratio |
|---|---|---|---|---|
| 1 | cold | 66,623 | 1,405 | 47.4x |
| 1 | warm | 84,526 | 1,466 | 57.7x |
| 32 | cold | 77,675 | 1,423 | 54.6x |
| 32 | warm | 101,107 | 1,465 | 69.0x |
| 128 | cold | 78,893 | 1,431 | 55.1x |
| 128 | warm | 100,456 | 1,461 | 68.8x |
| 1, emissions alone | warm | 107,084 | 7,589 | 14.1x |

Most of the parity numerics' time is the exact-sum CRF scorer; the packed
numerics' gain on the forward pass alone is the last row. The check that
sends a sentence with huge scores to the exact scorer costs nothing
measurable: against the code without it, three runs each on the same core,
32 sentences per call, the medians were 97,568 against 97,655 words per
second warm and 77,083 against 77,078 cold. The parity
numerics are as fast as before this path was added (an A/B over 400 test
sentences on the same core: within 1%).

### Bundle form

OpenNLP archive loading and `postag-convert` are retired under #1511.
Native producers call `turbo_bundle::postag::export` with an in-memory
`BilstmPosModel`, a typed `NativeSource { id, revision }`, a licence and a
new output directory. No archive is read. The version-2 bundle contains:

- `weights.safetensors`: every tensor as F32, row-major, by name,
  with LSTM gate rows in `(i, f, g, o)` order;
- `weights-f16.safetensors`: the same tensors as F16, for `FASTEST`;
- `vocab.json`: word, character (modified UTF-8 bytes) and pretrained
  vocabularies, tags and maximum word length;
- `postag.json`: native producer identity and revision, licence, gate
  order, CRF/stacked/adapter flags and each file's SHA-256.

`turbo-bundle postag-verify <dir>` checks native version-2 provenance and
file digests. It does not accept an archive comparison argument.
`turbo_bundle::postag::load` additionally validates tensor shapes, types
and model flags. Version-1 archive-derived manifests are not accepted.
`BilstmPosModel::named_tensors`, `vocabularies` and `from_named` expose the
native parameters in both directions.

Export checks every narrowed F32 and F16 weight for finiteness before
creating the output directory. Values that round to finite F16 remain
supported. File-system failures after directory creation can leave a
partial directory; the required manifest and digest checks reject it.

Bundle weights are rounded from native f64 parameters. Exact f64 score
or probability parity is not promised. Native regression tests compare
every tensor and vocabulary, then compare packed inference with an
explicitly F32-rounded model across optional-component variants. F16
boundaries, corruption, malformed tensors and retired CLI forms have
separate coverage. These tests establish format and inference contracts,
not trained-model accuracy.


### The turbo-bench record

`turbo-bench postag` measures the two numerics on a treebank and writes a
record to `benchmarks/records/postag/`: accuracy, agreement, emission and
marginal drift, and words per second and call latency by sentences per
call and by sentence length. The record on this model and the EWT test
split is `benchmarks/records/postag/x86-64.cpu.postag.bilstm-crf.819328e7.90e41ce15c8c.json`; docs/analysis.md ("BiLSTM-CRF
numerics") reads it.

### Not done

- Other backends. The packed numerics use the CPU backend's GEMM directly;
  no GPU or NPU kernel runs the tagger, so no `TURBO_TEST_REQUIRE_*`
  hardware run exists for it.
- A bundle import recipe built in a pinned container: there is no published
  BiLSTM-CRF model to import.
- The JVM on EWT. The EWT comparison is packed against the parity numerics,
  which the goldens tie to the JVM; the JVM was not run on the treebank.
