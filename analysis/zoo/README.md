# turbo-zoo

Model and dictionary catalog (`Zoo`, `Index`, `Component`, `Artifact`, `analysis/zoo/src/index.rs`) with SHA-256/SHA-512 content verification, local cache management, and licence/provenance enforcement for analysis models and lexical resources.

## Features

- **Typed model catalog (`Index`, `Component`, `Artifact`, `analysis/zoo/src/lib.rs`, `analysis/zoo/src/index.rs`, `analysis/zoo/src/index/component.rs`)**: Resolves model, lexicon, and dataset identifiers (`ArtifactId`) by task, language, and licence class (`Licence`, `LicenceClass`, `analysis/zoo/src/licence/mod.rs`) with mandatory cryptographic digest and provenance fields.
- **Content-addressed local cache (`Zoo`, `Home`, `analysis/zoo/src/home.rs`, `analysis/zoo/src/fetch.rs`, `analysis/zoo/src/digest.rs`)**: Verifies downloaded or local artifacts against expected digests (`Sha256`, `Sha512`) before handing out verified local files (`Fetched`, `analysis/zoo/src/transfer.rs`).
- **Licence gate (`Acceptances`, `Acceptance`, `analysis/zoo/src/accept.rs`)**: Refuses fetching non-default licence classes until explicitly accepted.

## Testing, Accuracy, and Speed

- **Test suite**: Verified by `analysis/zoo/tests/zoo.rs`, `analysis/zoo/tests/builtin.rs`, `analysis/zoo/tests/paths_and_cache.rs`, `analysis/zoo/tests/transfer.rs`, and `analysis/zoo/tests/cli.rs`. See `docs/zoo.md` for catalog schema details.
