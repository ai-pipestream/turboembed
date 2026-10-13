//! SciFact from BEIR: scientific claims, judged against the abstracts
//! that support or refute them. Read in the BEIR layout from the copy
//! MTEB publishes as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API; the files
//! could not be fetched from where this was written, so no digest is
//! recorded yet.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "scifact",
    license: "CC BY-SA 4.0, as the BEIR dataset card states it (the MTEB copy's card says unknown)",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/corpus.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/queries.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/qrels/test.tsv",
            sha256: Digest::NotRecorded,
        },
    ]),
    span_unit: None,
    read: beir::read,
};
