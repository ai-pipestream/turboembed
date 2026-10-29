//! Remote sources for the model zoo (#1613): a turboembed zoo server read
//! over gRPC, and the `TURBO_ZOO_SOURCE` list that names the places a
//! model loads from, the way DJL loads one from an `s3://` address.
//!
//! - [`ZooServer`]: a [`turbo_zoo::Transport`] that reads each file with
//!   the server's `ReadModelFile`, naming the artifact, the path and the
//!   SHA-256 the local index pins, so a server whose index pins other
//!   bytes refuses before sending any. The zoo verifies what arrives by
//!   size, SHA-256 and SHA-512, as it verifies a download from anywhere.
//! - [`S3Bucket`] (the `s3` feature, turbo-zoo's own): an S3 bucket or
//!   prefix, on AWS or an S3-compatible store such as RustFS, holding
//!   files in the same `blobs/sha256/<hex>` layout, read with one
//!   SigV4-signed GET each.
//! - [`sources`]: `file:///dir` (a content-addressed store),
//!   `grpc://host:port` and `grpcs://host:port` (a zoo server),
//!   `s3://bucket/prefix` and `origin` (the index's own addresses), tried
//!   in the order given. The first `file://` store also keeps what the
//!   sources after it serve.
//!
//! The licence check stays the local zoo's: no source is asked for an
//! artifact whose terms the user has not accepted, and a zoo server makes
//! its own check before it sends a byte.

mod server;
mod spec;

pub use server::ZooServer;
pub use spec::{ENV_SOURCE, sources};
#[cfg(feature = "s3")]
pub use turbo_zoo::{S3Bucket, sigv4};

/// The zoo server's messages and client, generated from
/// server/proto/turbo_training.proto.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("turbo.training.v1");
}
