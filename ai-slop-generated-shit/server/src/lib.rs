//! Inferstream's server surfaces as a library, so the binary and the tests
//! drive exactly the same code: [`config`] parses what to serve, [`engine`]
//! loads it and runs the Turbo tasks, [`oip`] is the Open Inference
//! Protocol v2 mapping both bindings share, [`http`] is the axum router
//! (OIP REST, the OpenAI-shaped routes, `/info`) and [`grpc`] is
//! `inference.GRPCInferenceService`. The binary is `src/main.rs`.

#![deny(missing_docs)]

pub mod config;
pub mod engine;
pub mod error;
pub mod grpc;
pub mod http;
pub mod oip;
