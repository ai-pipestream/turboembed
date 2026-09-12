The wire-contract `.proto` files live at the repo root:

    proto/open_inference_grpc.proto
    proto/inferstream_extension.proto

`crates/protocol/build.rs` compiles those files. This directory is kept so
older docs that mention `crates/protocol/proto/` still resolve; do not add
a second copy of the contract here.
