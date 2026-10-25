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

The current existing-path samples, compiler, source and corpus hashes,
commits, run order and ambient load are in
`../../reference/existing-path-timing.json`. They compare main
`bf52319dfc49486e5109f2c76faccb8fcd6edc02` with candidate
`89c2ed766746fe3edf801f744b1b95713057c40e` on an AMD Ryzen 9 9950X3D.
The candidate adds only an optional match dump to the example after timing;
its matcher code is unchanged from the reviewed implementation.

Twelve alternating pairs give process medians of 13.439 ms for main and
13.514 ms for the candidate (ratio 1.0056). The median paired ratio is
0.9894. Individual ratios range from 0.7757 to 1.3104 on the shared host,
whose one-minute load moved from 40.28 to 38.90. These samples are
consistent with timing parity within the observed variation; no speedup
or isolated-host result is claimed. All twelve pairs are retained.

To repeat with the recorded rustc 1.99.0 toolchain, set `BASE` and
`CANDIDATE` to absolute clean checkouts at those commits, and `CORPUS` to
the digest-checked EWT train file. The same example is linked to each
checkout's release libraries, keeping both source trees clean:

```bash
for checkout in "$BASE" "$CANDIDATE"; do
  out="$checkout/target/matcher-timing"
  (cd "$checkout" && CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$out" \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --locked --release -p turbo-matcher --lib)
  analysis=("$out"/release/deps/libturbo_analysis-*.rlib)
  layers=("$out"/release/deps/libturbo_analysis_layers-*.rlib)
  test "${#analysis[@]}" = 1 && test "${#layers[@]}" = 1 || exit 1
  rustc --edition=2024 -C opt-level=3 -C target-cpu=native \
    -L "dependency=$out/release/deps" \
    --extern "turbo_matcher=$out/release/libturbo_matcher.rlib" \
    --extern "turbo_analysis=${analysis[0]}" \
    --extern "turbo_analysis_layers=${layers[0]}" \
    "$CANDIDATE/analysis/matcher/examples/token_bench.rs" \
    -o "$out/token_bench"
done
base="$BASE/target/matcher-timing/token_bench"
candidate="$CANDIDATE/target/matcher-timing/token_bench"
taskset -c 7 "$base" "$CORPUS" 1 "$BASE/target/base-matches.txt"
taskset -c 7 "$candidate" "$CORPUS" 1 "$CANDIDATE/target/candidate-matches.txt"
cmp "$BASE/target/base-matches.txt" "$CANDIDATE/target/candidate-matches.txt"
for ((pair=0; pair<12; pair++)); do
  if ((pair % 2 == 0)); then order=("$base" "$candidate");
  else order=("$candidate" "$base"); fi
  for binary in "${order[@]}"; do taskset -c 7 "$binary" "$CORPUS" 15; done
done
```

Each process performs one warmup, reports 204,577 tokens and 51,765 hits,
and prints the median of fifteen timed calls. The optional dump is an
additional untimed call. Both complete match dumps are byte-identical
(4,167,221 bytes), SHA-256
`af63717117339df6cfedc46020317acd2b4b6b17ce0f052f37d4583174335ffd`.
The receipt/doc follow-up does not change the measured matcher or example.
