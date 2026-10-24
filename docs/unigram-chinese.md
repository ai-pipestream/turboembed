# Chinese unigram modes and evaluation

`UnigramSegmenter::new` now defaults to `UnigramMode::Dictionary`: ASCII
letters and digits form maximal runs, including mixed runs such as `abc123`.
The dictionary path and tie rules are otherwise unchanged. The optional
`UnigramMode::Chinese` adds jieba's trained BMES unknown-word model;
`UnigramMode::OpenNlp` preserves the previous behavior, including splitting
unlisted ASCII characters, for JVM goldens and differential fuzzing.
Use `.with_mode(UnigramMode::Chinese)` or `.with_mode(UnigramMode::OpenNlp)`
to select those policies.

Chinese mode runs Viterbi on contiguous single-character Han words from
the dictionary path, unless the entire run is itself a known word. Listed
multi-character words remain boundaries. Punctuation, other scripts and
characters absent from the HMM break the run. The decoder uses shared
Unicode script and whitespace properties and no regular expressions.
ASCII runs are atomic before dictionary decoding, so a mixed-script
lexicon entry crossing such a run is split in the two new modes.

All offsets refer to the unchanged input view. Sentence ownership,
UTF-8/UTF-16 round trips, append behavior, whitespace gaps, supplementary
Han and emoji are tested. Each policy has a distinct configuration hash;
Chinese mode includes the SHA-256 of the HMM. The lexicon remains the
annotator's model reference. OpenNLP mode retains its previous hash.
The [model source and generator](../analysis/tokenize/data/jieba-hmm/README.md)
include the MIT licence and input checksums. This adds no workspace dependency.

## SIGHAN accuracy

Measured 2026-10-24 using the complete jieba dictionary at
`67fa2e36e72f69d9134b8a1037b83fbb070b9775` (349,046 lines) and the original
SIGHAN 2005 release 1 PKU/MSR test and gold files. PKU has 1,945 input lines,
MSR 3,985. The scorer checks exact word boundaries over the concatenated
characters, accounting for the different MSR line breaks, and asserts that
predicted and gold text are identical after whitespace removal. OOV recall
uses the release's training-word lists. The corpus remains outside the
repository and is used only for this bakeoff evaluation.

| Corpus | Mode | Precision | Recall | F1 | Correct / returned / gold | OOV found / total |
|---|---|---:|---:|---:|---|---|
| PKU | OpenNlp | 0.824292061 | 0.807122600 | 0.815616982 | 84241 / 102198 / 104372 | 2519 / 6006 |
| PKU | Dictionary (default) | 0.858555755 | 0.814250949 | 0.835816639 | 84985 / 98986 / 104372 | 3262 / 6006 |
| PKU | Chinese (BMES) | 0.850614113 | 0.786963937 | 0.817552032 | 82137 / 96562 / 104372 | 3481 / 6006 |
| PKU | jieba-rs without HMM | 0.858600554 | 0.814260530 | 0.835842914 | 84986 / 98982 / 104372 | 3264 / 6006 |
| PKU | jieba-rs with HMM | 0.852800947 | 0.786609435 | 0.818368944 | 82100 / 96271 / 104372 | 3500 / 6006 |
| MSR | OpenNlp | 0.820005155 | 0.833578172 | 0.826735958 | 89087 / 108642 / 106873 | 955 / 2829 |
| MSR | Dictionary (default) | 0.820005155 | 0.833578172 | 0.826735958 | 89087 / 108642 / 106873 | 955 / 2829 |
| MSR | Chinese (BMES) | 0.817131722 | 0.812169584 | 0.814643097 | 86799 / 106224 / 106873 | 1271 / 2829 |
| MSR | jieba-rs without HMM | 0.820005155 | 0.833578172 | 0.826735958 | 89087 / 108642 / 106873 | 955 / 2829 |
| MSR | jieba-rs with HMM | 0.816289985 | 0.810401130 | 0.813334898 | 86610 / 106102 / 106873 | 1271 / 2829 |

Grouped dictionary mode raises PKU F1 from 0.815617 to 0.835817 and leaves
MSR unchanged at 0.826736. BMES increases OOV recall but lowers whole-corpus
F1 on both sets relative to grouped dictionary mode. This is why it is an
explicit option rather than the default. The new modes do not claim exact
jieba parity: forward dictionary ties remain OpenNLP's, ASCII runs are
atomic, and HMM eligibility follows the shared Han script property and
observed model characters, rather than jieba's block matching rules.

## Speed

AMD Ryzen 9 9950X3D, Linux x86-64, Rust 1.99.0, release profile, one pinned
logical CPU, 2026-10-24. The dictionary and HMM are loaded before timing.
One untimed warmup pass and eight recorded passes each process the whole
corpus 50 times; implementation order reverses on alternating passes.
Times below are the median milliseconds per complete corpus, with the
50 repetitions averaged within each pass. Reading, scoring and model
construction are excluded. Turbo reuses span and scratch buffers;
jieba-rs returns its normal allocated token vector. This measures the
public segmentation paths, not equal allocator behavior or startup cost.

| Corpus | Mode | Median ms |
|---|---|---:|
| PKU | OpenNlp | 3.393527 |
| PKU | Dictionary (default) | 3.657739 |
| PKU | Chinese (BMES) | 4.473805 |
| PKU | jieba-rs without HMM | 4.226767 |
| PKU | jieba-rs with HMM | 4.398955 |
| MSR | OpenNlp | 3.748905 |
| MSR | Dictionary (default) | 4.025402 |
| MSR | Chinese (BMES) | 4.939541 |
| MSR | jieba-rs without HMM | 4.578814 |
| MSR | jieba-rs with HMM | 4.716219 |

The default is faster than jieba-rs without HMM in this run. The BMES mode
is slightly slower than jieba-rs with HMM. These are host-specific timings,
not a throughput guarantee. Eight per-pass values, in milliseconds:

```text
pku OpenNlp: 3.423816, 3.366349, 3.411678, 3.354231, 3.426417, 3.366257, 3.411309, 3.375745
pku Dictionary (default): 3.644682, 3.628422, 3.664342, 3.660865, 3.670135, 3.670794, 3.654613, 3.651022
pku Chinese (BMES): 4.456200, 4.435294, 4.471290, 4.463729, 4.481841, 4.476320, 4.502668, 4.493825
pku jieba-rs without HMM: 4.172029, 4.513133, 4.205296, 4.250620, 4.184513, 4.248239, 4.190096, 4.255172
pku jieba-rs with HMM: 4.380921, 4.397756, 4.385482, 4.416588, 4.400155, 4.447414, 4.438074, 4.389988
msr OpenNlp: 3.733581, 3.733176, 4.150462, 3.769320, 3.767323, 3.682820, 3.764228, 3.716416
msr Dictionary (default): 4.026764, 4.085646, 4.412223, 3.999618, 4.006987, 4.016621, 4.031961, 4.024039
msr Chinese (BMES): 4.935269, 4.973110, 4.934966, 4.943813, 4.917814, 4.907122, 4.969048, 5.052650
msr jieba-rs without HMM: 4.573899, 4.605562, 4.517818, 4.583729, 4.528827, 4.781332, 4.521852, 4.866359
msr jieba-rs with HMM: 4.681131, 4.803852, 4.679871, 4.719884, 4.712554, 5.014440, 4.671330, 4.934248
```

## Reproduce

The ignored test checks all three Turbo modes against the exact counts above:

```sh
TURBO_JIEBA_DICT="$DICT" TURBO_SIGHAN_DIR="$CORPUS" \
  cargo test --locked --release -p turbo-tokenize --test unigram_sighan -- --ignored --nocapture
```

`DICT` identifies the pinned dictionary file; `CORPUS` identifies the extracted
`icwb2-data` directory. The test never fetches either resource automatically.
For the comparison, copy
[the comparison program](../analysis/tokenize/comparisons/jieba.rs) into a
separate Cargo binary's main source file. Its manifest needs `jieba-rs = "=0.11.0"`
and a path dependency on this checkout's `analysis/tokenize`. Keep this
comparison outside the workspace, generate its lockfile once, then build
with `cargo build --locked --release`. Run the resulting executable with
`DICT` and `CORPUS` as its two arguments, pinned to one CPU. An optional
third argument `samples` runs only the four printed examples. No comparator
dependency is added to the product or its tests.

Input SHA-256 values:

```text
7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8  dict.txt
296c54adf4abc3702eca0cbce70b55fe22a344ea02d33175a4d1183e074072b4  icwb2-data.zip
48c2655b535ea33802c873373f3176e57d39ba1a45a4dbba164e9125d7ce149e  icwb2-data/testing/pku_test.utf8
8068b0c5a8f309e2ff2ba88dc26de7a8dac2f2684c9c7d8094337d9ce272f601  icwb2-data/testing/msr_test.utf8
913f78b20b17ea1e154f6246644d7d624b2710641f109a15daee9d63c9fb88d4  icwb2-data/gold/pku_test_gold.utf8
cd1a8473841f1b2fcddd14d12599ad8872e6167feb64807af5bac2f6a32cb75d  icwb2-data/gold/msr_test_gold.utf8
68fdbcef065d315e5dc3dc4c0e1b68997b1849141ba93b8fa2325fb088b5b0f3  icwb2-data/gold/pku_training_words.utf8
d5328d5cc8576c8e008e70ad33882aae4ce2cbbbf6cd1130c59a66a248961b8c  icwb2-data/gold/msr_training_words.utf8
```
