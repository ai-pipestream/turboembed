# spaCy pattern oracle

`drugs_patterns.jsonl` is the complete, unmodified 118-rule file from
[Explosion projects](https://github.com/explosion/projects/blob/504e1e428238a08fb038c04af04ea4b3b34756af/tutorials/ner_drugs/assets/drugs_patterns.jsonl).
Its MIT licence is `projects-LICENSE`. SHA-256:
`59ef77c3cd0a0a64b9794346b9d83fc9e2bfcbb6fb4f68b09de92c828ed56457`.

`operators.jsonl` adds project-authored rules for the additional operators.
The optional local oracle records spaCy 3.8.7 output on 31 real sentences, 485 tokens
and their gold annotations from
[UD English EWT r2.16](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.16).
EWT is CC BY-SA 4.0, which also applies to extracted sentences and
annotations. The corpus and generated oracle stay local and are never
checked in, following `docs/zoo.md`. Read the upstream
[licence](https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/LICENSE.txt)
before downloading or using them.
The train file SHA-256 is
`3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b`.

The selection, tokenization constraints and generator are in
`../../reference/spacy_patterns.py`. It selects every reconstructable
sentence matching a published drug rule, twenty short sentences, and ten
short sentences matching fuzzy operators. It compares all matches,
including non-matches, not only the published pattern hits. There are 2,580
matches (two from the published drug rules). The fixture preserves original
text and UTF-8 byte offsets. No model runs in place of the gold annotations.

Regeneration uses the repository's reference container image
`sha256:8d07bd26184d6efe28889ca6a78e3d672f4eaa31a14930f1472cad243476914d`
with spaCy 3.8.7 installed separately for the oracle; it adds no Rust
dependency. Run the generator with the EWT train file, this directory, and
an output JSON filename. Pin the image and spaCy version when repeating.

After accepting the upstream licence, download the r2.16 train file from
its original repository into `target/spacy/en_ewt-ud-train.conllu` and
verify the SHA-256 above. The generator's arguments are:

```sh
python analysis/matcher/reference/spacy_patterns.py \
  target/spacy/en_ewt-ud-train.conllu analysis/matcher/tests/data \
  target/spacy/oracle.json
```

Run that command inside the pinned reference container with spaCy 3.8.7,
mounting the input and output directories. Then run from the repository root:

```sh
TURBO_SPACY_ORACLE="$PWD/target/spacy/oracle.json" \
TURBO_TEST_REQUIRE_SPACY_ORACLE=1 \
cargo test --locked -p turbo-matcher --test import_spacy
```

Without `TURBO_SPACY_ORACLE`, the external comparison reports that it is
skipped; the import, operator and pipeline tests still run. Setting
`TURBO_TEST_REQUIRE_SPACY_ORACLE=1` makes the missing path fail. An explicit
path that cannot be read, invalid JSON, a different corpus digest or a
different spaCy version always fails. The test pins 31 sentences, 485
tokens and all 2,580 expected matches.

The historical existing-path timing samples, source and corpus hashes,
commits and run order are in `../../reference/existing-path-timing.json`.
To repeat, build the committed `examples/token_bench.rs` in each named
checkout with `cargo build --locked --release -p turbo-matcher --example token_bench`
(copy the same example into the baseline checkout first). For twelve pairs,
alternate the order of the two binaries and run each as
`taskset -c 4 target/release/examples/token_bench CORPUS.conllu 15`.
Each process performs one warmup, reports 204,577 tokens and 51,765 hits,
and prints the median of fifteen samples. Each checkout needs its own
target directory. These retained samples are from the original run;
the oracle-storage correction does not claim a new measurement.
