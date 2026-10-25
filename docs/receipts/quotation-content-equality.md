# Quotation content equality

Follow-up to #1244, part of #1202.

Base: `df3f37e0f8adf58319723116857ddbddc6145b2b`.
Regression-only commit: `01da2c597882763b0102f56b8529740fed4f8d02`.
Corrected source: `4c137327a1ebf1256073a15a4a2f6455d5ba7f1d`.

`Quotations::eq` previously compared content envelopes and row fields,
omitting the fragments inside each envelope. On `é-x-é`, continuous
content `0..7` compared equal to discontinuous content `0..2, 5..7`.
Both records pass native layer validation and document insertion.
The comparison now also checks each row's actual fragments without
allocating. Existing field, view and floating-point equality remain.

The permanent `quotation_equality` integration target reproduces three
failures before the fix, with one passing control. Afterward all four
tests pass. They distinguish different gaps with equal envelopes,
check later rows, preserve named records through real binary graph
export/import, and retain optional-field and signed-zero behavior.
The round-trip test checks fragments explicitly as well as equality.

Validation with two build jobs, debug information and incremental builds
disabled:

- `cargo test --locked -p turbo-analysis-layers -p turbo-coref`:
  369 passed, 0 failed, 0 ignored across 41 test/doc-test targets.
- `cargo clippy --locked -p turbo-analysis-layers -p turbo-coref --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `cargo test --locked -p turbo-bench --test tree`: 8 passed.

These are native correctness tests, not model-accuracy or performance
measurements. This repair does not complete #1202's broader acceptance.
