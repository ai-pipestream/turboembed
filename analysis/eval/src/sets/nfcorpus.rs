//! NFCorpus from BEIR: nutrition questions, judged against medical
//! articles with graded relevance. Read in the BEIR layout from the copy
//! MTEB publishes as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API. Each
//! digest is the SHA-256 of two separate downloads of the file, which
//! agreed.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "nfcorpus",
    license: "CC BY-SA 4.0, as the BEIR dataset card states it (the MTEB copy's card states none)",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/corpus.jsonl",
            sha256: Digest::Recorded("ee3db77bc2183e66b9c114e6eca4adf44de381705b1f8c94de8efdd232f0c4ff"),
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/queries.jsonl",
            sha256: Digest::Recorded("1b893f53f3f62e8f2949a6db619a9cb058e8d97901a5010e108e60c4ef5fd924"),
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/qrels/test.tsv",
            sha256: Digest::Recorded("f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a"),
        },
    ]),
    span_unit: None,
    read: beir::read_in,
};
