# Geo mention admission

Base: `272b9298`. Fixture: `Dublin’s c C S Sao Paulo Cordoba`.
The actual bundled gazetteer resolves all 7 supplied spans under the
existing policy. With `ExcludeSingleAsciiLetters`, only Dublin, Sao Paulo
and Cordoba produce locations. The original 7 entities remain available.

The regression first failed with the default policy: expected 3 locations,
got 7. Enabling the new policy passes. Additional tests check that filtering
happens before geocoder context, non-ASCII names remain eligible, invalid
UTF-8/out-of-range spans return errors, and configuration hashes distinguish
policies and entity type sets.

Validation, CPU:

```sh
cargo fmt --all --check
cargo test -p turbo-geo
cargo clippy -p turbo-geo --all-targets -- -D warnings
```

All passed: 94 tests, no failures or skips, including the existing JVM
parity suite. This is a filter for the demonstrated false links, not a
measure of NER or contextual disambiguation accuracy. The default policy
and bundled table are unchanged.

Review focus: opt-in semantics, legitimate single-letter names, filtering
before spatial context, bounds validation and provenance.
