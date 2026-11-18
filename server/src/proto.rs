//! Generated gRPC message and service definitions for `turbo-kserve`.

#![allow(clippy::all)]

tonic::include_proto!("inference");

/// The analysis result layer (docs/grpc-analysis.md).
pub mod result {
    pub mod v1 {
        tonic::include_proto!("turbo.result.v1");
    }
    pub use v1::*;
}

/// Hybrid search, index lifecycle, aliases, and collection drift (issue #1221).
pub mod search {
    pub mod v1 {
        tonic::include_proto!("turbo.search.v1");
    }
    pub use v1::*;
}

/// The gRPC health checking protocol (`grpc.health.v1.Health`).
pub mod health {
    tonic::include_proto!("grpc.health.v1");
}

/// The downstream document sink (docs/grpc-sink.md).
pub mod sink {
    pub mod v1 {
        tonic::include_proto!("turbo.sink.v1");
    }
    pub use v1::*;
}

/// The acknowledged PII access audit (docs/grpc-analysis.md, PII value
/// release).
pub mod audit {
    pub mod v1 {
        tonic::include_proto!("turbo.audit.v1");
    }
    pub use v1::*;
}

/// The vocabulary store's service (docs/grpc-vocabulary.md).
pub mod vocabulary {
    pub mod v1 {
        tonic::include_proto!("turbo.vocabulary.v1");
    }
    pub use v1::*;
}

/// The model catalog and static model training (docs/grpc-catalog.md,
/// docs/grpc-training.md).
pub mod training {
    pub mod v1 {
        tonic::include_proto!("turbo.training.v1");
    }
    pub use v1::*;
}
