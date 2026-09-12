# LEGACY: `libMlxEngine.dylib` C ABI

This package is **not** the Apple serve path anymore.

The supported Mac server is the all-Swift gRPC binary in `swift/`:
it folds the same mlx-swift / mlx-swift-lm engine **in-process** and
implements `inference.GRPCInferenceService` + `inferstream.v1` with
grpc-swift. There is no Rust façade and no C ABI on that path.

This directory is kept so the legacy Rust `inferstream-apple` binary
(`crates/arch-apple` + `crates/backend-apple`) still links for CI
type-checking. Do not add features here expecting them to ship on Mac.
