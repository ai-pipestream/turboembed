# Contributing to inferstream

Thanks for your interest! This is an early-stage project; the fastest way to
help is small, focused changes with tests.

## Development setup

- Rust ≥ 1.85 (`rustup default stable`)
- `protoc` on `PATH` (`apt install protobuf-compiler` or a
  [release binary](https://github.com/protocolbuffers/protobuf/releases))

```bash
cargo build --workspace
cargo test --workspace          # default features; must stay green on Linux
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
```

## Ground rules

- **Default features must build and test on Linux with no engine or Apple
  dependencies.** Engine backends are wired only in the arch binaries
  (`crates/arch-nvidia`, `crates/arch-intel`, `crates/arch-apple`); the
  shared `crates/server` stays engine-free. Real GPU runtime links go behind
  opt-in features (e.g. `trtllm-sys`) that CI never enables.
- The vendored proto in `crates/protocol/proto/` tracks upstream
  [kserve/open-inference-protocol](https://github.com/kserve/open-inference-protocol);
  local extensions must be clearly marked `INFERSTREAM EXTENSION` and kept
  wire-compatible with Triton conventions where one exists.
- Prefer raw tensor payloads (`raw_input_contents` / `raw_output_contents`)
  in new code paths; the repeated scalar fields exist for spec compliance,
  not as the primary path.
- New backends implement the `Backend` trait in `crates/backend`; do not add
  backend-specific branches to the server service layer.
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`,
  `test:`).

## Adding a backend (sketch)

1. Create `crates/backend-<name>` implementing `inferstream_backend::Backend`.
2. Add a `BackendKind` variant (and any typed config fields) in
   `crates/server/src/config.rs`.
3. Wire construction in the factory of the arch binary that should ship it
   (`crates/arch-*/src/main.rs`), behind a cargo feature if it pulls native
   deps.
4. Add tests against the trait; only integration-test with real models behind
   an opt-in feature or ignored test.

## Reporting issues

Include the config (redact keys), the exact RPC and payload shape, and
server logs at `RUST_LOG=debug`.
