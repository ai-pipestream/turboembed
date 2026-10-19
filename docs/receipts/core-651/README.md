# Core boundary and split receipts

Base: `103520524d08addfc000c91c79f351e7ea27705b`.

## Reproductions

The first regressions ran after the mechanical source split, before behavioral
fixes. `red-formats.log` records six failing tests: trailing manifest JSON, NUL
inside C header strings, mean timing outside its observed range, duplicate
reference names, unrepresentable derived speed ratios, and invalid inputs to
the public speed helpers. `red-unigram.log` records six more failures:
malformed normalizers and boolean fields, invalid direct-call metaspace,
noncanonical base64 padding, invalid replacement data, and lattice overflow.

For the lattice case, finite scores scaled by 1e308 changed the best encoding
of `aba` from `[3, 1]` to `[1, 4]`. Scaling the lattice only when accumulation
could overflow restores `[3, 1]`; ordinary score arithmetic is unchanged.

`red-decision.log` records a further failure: an unchecked record with NaN
conformance still produced a Supported capability verdict. Capability decisions
now validate the record. An existing wider-batch test fixture was corrected to
update its computed token count along with its batch size.

All 13 new regression tests pass: seven in `green-formats.log`, six alongside
three existing unigram tests in `green-unigram.log`.

## Gates

All required commands passed:

- `cargo fmt --all --check`: `fmt.log` (empty on success).
- `cargo clippy --locked -p turbo -p turbo-bench --all-targets -- -D warnings`: `clippy.log`.
- `cargo test --locked -p turbo`: `tests.log`, 314 passed, 10 existing tests ignored.
- `cargo test --locked -p turbo-bench --test tree`: `tree.log`, 3 passed.

Core tests used the isolated, checkout-relative `target/core` build directory
specified by README. Logs replace the checkout prefix with `.`. These are CPU
checks; no device backend validation is claimed.

## Split and class table checks

Every touched Rust file is below 500 lines; `line-counts.txt` lists all 33
files (largest: 481 lines). Public module exports are retained. No CPU
implementation files or benchmark data records changed.

The class generator now writes the split table files. Its existing ignored test
entry point was run explicitly (`generator.log`, 1 passed). All 821 regenerated
letter and number ranges match the original table in order and value. The full
suite includes scalar/range invariants and exhaustive comparison of character
classification with the existing development-only regex oracle. No runtime
regex dependency was added. The BPE tests were moved into their own file so
updating that generator entry point also meets the 500-line limit.
