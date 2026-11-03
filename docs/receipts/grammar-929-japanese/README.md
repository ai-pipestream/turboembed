# Existing English grammar comparison

This bounded comparison checks the existing English selection path after
adding the optional CJK word-boundary predicate. The source sample is the
repository's existing 102456-byte English sample. The harness parses it and
selects Numbers, Ordinals, Percents and Quantities. Complete serialized results
match byte-for-byte for both grammars, not only the reported counts.

- Main: `75160eceda1101837cb0dbfe23b435b591b9ca8b`.
- Initial candidate: `5a66e0c956fa6ed526d7f8b893a0d9a83528fffe`.
- Revised candidate: `783b90d20f6ee32d9fb494538e595a385d2adb2e`.
- Rust 1.99.0, default Cargo release profile, build jobs 2.
- Eight pairs with alternating execution order; each process warms three
  iterations, then times twenty parse-and-select iterations per grammar.
- This was a shared host. Other work was active. CPU identity, load samples,
  binary SHA256, harness/sample SHA256, output SHA256 and raw stdout are in
  both JSON receipts. These are not isolated-host speed measurements.

| Grammar | Initial paired candidate/main | Revised main median ms | Revised candidate median ms | Revised paired candidate/main |
| --- | ---: | ---: | ---: | ---: |
| English numbers | 1.057779 | 6.446767 | 5.924341 | 0.969296 |
| English units | 1.066279 | 9.429205 | 8.874477 | 0.920354 |

The first run suggested overhead. The revised implementation obtains an
optional predicate once per selection, avoiding repeated virtual calls on
candidates for grammars that retain the default boundary rule. Correctness
remained unchanged. The final run did not repeat the initial slowdown; the
change in baseline absolute timings also shows why these runs do not establish
an isolated speedup. Both receipts are retained rather than replacing the
initial observation.

To reproduce, copy `existing_path.rs` to
`analysis/grammar/examples/cjk_existing_path.rs` in each exact checkout, build
with `cargo build --release -p turbo-grammar --example cjk_existing_path -j2`
and distinct `CARGO_TARGET_DIR` values, then run from the revised checkout:

```sh
python3 docs/receipts/grammar-929-japanese/compare.py \
  "$MAIN_BINARY" "$CANDIDATE_BINARY" \
  75160eceda1101837cb0dbfe23b435b591b9ca8b \
  783b90d20f6ee32d9fb494538e595a385d2adb2e \
  target/english-comparison.json
```

The script stores complete output proofs under `target/proof` and asserts
equality before starting the bounded comparison. No repeat is automatic.
