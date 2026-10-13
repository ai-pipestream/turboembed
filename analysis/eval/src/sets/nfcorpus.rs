//! NFCorpus from BEIR: nutrition questions, judged against medical
//! articles with graded relevance. Read in the BEIR layout from the copy
//! MTEB publishes as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API; the files
//! could not be fetched from where this was written, so no digest is
//! recorded yet.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "nfcorpus",
    license: "CC BY-SA 4.0, as the BEIR dataset card states it (the MTEB copy's card states none)",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/corpus.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/queries.jsonl",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/nfcorpus/resolve/52ac3f19d3449632d9f00aab0ad34a110fc03816/qrels/test.tsv",
            sha256: Digest::NotRecorded,
        },
    ]),
    span_unit: None,
    read: beir::read,
};
