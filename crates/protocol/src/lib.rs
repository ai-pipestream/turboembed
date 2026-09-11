//! Vendored KServe Open Inference Protocol (OIP) V2 gRPC types for
//! inferstream, plus tensor wire-format helpers.
//!
//! The proto source lives in `proto/open_inference_grpc.proto`, vendored from
//! [kserve/open-inference-protocol] at commit
//! `d49cc23f89d709d87b210ef9449e273ae243984e`, with a clearly marked
//! Triton-shaped `ModelStreamInfer` extension for bidirectional streaming.
//!
//! [kserve/open-inference-protocol]: https://github.com/kserve/open-inference-protocol

// TensorError carries the offending dtype/shape for diagnostics, which makes
// the Err variant larger than clippy's default threshold; these paths are not
// hot enough to justify boxing.
#![allow(clippy::result_large_err)]

/// Generated protobuf/gRPC types for the `inference` package.
pub mod inference {
    tonic::include_proto!("inference");
}

pub mod tensor;

pub use tensor::{DataType, TensorError};
