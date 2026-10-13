//! ArguAna from BEIR: arguments, judged against their best
//! counterargument. Read in the BEIR layout from the copy MTEB publishes
//! as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API; the files
//! could not be fetched from where this was written, so no digest is
//! recorded yet.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "arguana",
    license: "CC BY-SA 4.0, as the MTEB and BEIR dataset cards state it",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/corpus.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/queries.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/arguana/resolve/6c1bcf74b13dfd823aff056b79d4d93e702f19c7/qrels/test.tsv",
            sha256: Digest::NotRecorded,
        },
    ]),
    span_unit: None,
    read: beir::read,
};
