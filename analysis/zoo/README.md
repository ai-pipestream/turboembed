# turbo-zoo

Model and dictionary catalog (`ZooIndex`, `ZooComponent`, `analysis/zoo/src/index.rs`) with SHA-256/SHA-512 content verification, local cache management, and licence/provenance enforcement for analysis models and lexical resources.

## Features

- **Typed model catalog (`analysis/zoo/src/lib.rs`, `analysis/zoo/src/index.rs`, `analysis/zoo/src/index/component.rs`)**: Resolves model, lexicon, and dataset identifiers by task, language, and licence class (`analysis/zoo/src/licence/mod.rs`) with mandatory cryptographic digest and provenance fields.
- **Content-addressed local cache (`analysis/zoo/src/home.rs`, `analysis/zoo/src/fetch.rs`, `analysis/zoo/src/digest.rs`)**: Verifies downloaded or local artifacts against expected digests before moving them into the local cache (`analysis/zoo/src/transfer.rs`).
- **Licence gate (`analysis/zoo/src/accept.rs`)**: Refuses fetching non-default licence classes until explicitly accepted.

## Testing, Accuracy, and Speed

- **Test suite**: Verified by `analysis/zoo/tests/zoo.rs`, `analysis/zoo/tests/builtin.rs`, `analysis/zoo/tests/paths_and_cache.rs`, `analysis/zoo/tests/transfer.rs`, and `analysis/zoo/tests/cli.rs`. See `docs/zoo.md` for catalog schema details.
