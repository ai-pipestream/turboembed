//! ArguAna from BEIR: arguments, judged against their best
//! counterargument. Read in the BEIR layout from the copy MTEB publishes
//! as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API. Each
//! digest is the SHA-256 of two separate downloads of the file, which
//! agreed.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "arguana",
    license: "CC BY-SA 4.0, as the MTEB and BEIR dataset cards state it",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/corpus.jsonl",
            sha256: Digest::Recorded("c5cb4da0f464e93f19577bf4cfb18e9833c3725da12631eb53554628013b0aa9"),
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/queries.jsonl",
            sha256: Digest::Recorded("963c899dd307c9d32416ef47d0701f0be778daeac610af71d735df0d789371a4"),
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/qrels/test.tsv",
            sha256: Digest::Recorded("0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76"),
        },
    ]),
    span_unit: None,
    read: beir::read,
};
