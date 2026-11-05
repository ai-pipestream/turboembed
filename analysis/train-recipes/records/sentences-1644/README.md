# Native rule sentence measurements

Part of #1644. Executed 2026-11-05 using the offline evaluator at
`8627af5e16dd9bb04c62789b0ff3242f554d2284` on main
`9ec590641fa51c4214067f7ffe894151bc52aa14`. These are exact-span gold-label
measurements, not model-parity, device-throughput or trained-head results.

| Test set | Groups | Gold spans | Correct | Predicted | Rule F1 (%) | Newline F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD English EWT r2.18 | 208 | 2,077 | 948 | 1,301 | 56.127886 | 0.000000 |
| UD Spanish AnCora r2.16 | 173 | 1,721 | 1,585 | 1,644 | 94.205052 | 0.105597 |
| MASC 3.0.0 held-out | 639 | 6,385 | 4,316 | 5,127 | 74.982627 | 0.131839 |

The complete records retain per-group counts, corpus and sample digests,
candidate configuration and executable SHA-256, and deterministic paired
bootstrap intervals. All three commands returned 0: the existing gate compares
against the newline baseline. That result does **not** establish an acceptable
absolute quality floor or superiority over SaT, spaCy or Stanza. EWT and MASC
remain substantial accuracy gaps to investigate; no rule tuning used these
held-out results.

## Protocol and provenance

This command deliberately retains the existing gold-gate reader protocol:
groups of up to ten sentence texts with reconstructed separators. UD samples
are not original document reconstructions. MASC uses its recorded held-out
documents with Twitter excluded, then the existing sentence reader. The
bootstrap units are these groups, not whole documents. These scores must not
be compared directly to a document-level or differently grouped peer run.
The near-zero newline baseline reflects this grouping and is a weak comparator.
SaT, native-head and peer runs must use exactly the same input construction or
publish a separately identified document-level protocol for all systems.

Inputs were already available locally. No corpus text, weights or credentials
are published here; only aggregate counts and hashes. EWT r2.18 is CC BY-SA 4.0
and remains caller-supplied/accept-first. Spanish AnCora r2.16 is CC BY 4.0;
MASC 3.0.0 is CC BY 3.0 US. Existing recipe attribution and terms remain in
`trainer/recipes/src/ud.rs`, `trainer/recipes/src/ud_banks.rs` and
`analysis/train-recipes/src/masc.rs`. No licence tier changed.

The EWT test SHA-256 matched the recipe pin
`fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a`;
Spanish AnCora matched
`2a24316503a56e171f07b38cf2ec13c22e3b62a337c6dbf2d9d1a54ada2b012b`.
The MASC command checked the complete recorded corpus digest before reading
the fixed split. Each JSON record includes the exact corpus digest it scored.

## Reproduce

Build with Rust 1.99, `cargo build --locked -p turbo-train-recipes --bin
turbo-sentence-eval`. Invoke that binary as described in
[the evaluator documentation](../../SENTENCE-EVALUATION.md), using language
`en` and set `UD-EWT-r2.18/test`, or language `es` and set
`UD-Spanish-AnCora-r2.16/test`. Use the two pinned digests above. The MASC
subcommand accepts the existing pinned corpus root. Build profile was debug;
there is no timing claim. A rebuild can change the executable digest without
changing scores; do not rewrite the recorded measured binary identity.

SaT, native-head and spaCy/Stanza measurements remain outstanding. #1644
stays open. A passing fixture suite or this rule-vs-newline gate cannot close it.
