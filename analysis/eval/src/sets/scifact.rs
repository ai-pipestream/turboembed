//! SciFact from BEIR: scientific claims, judged against the abstracts
//! that support or refute them. Read in the BEIR layout from the copy
//! MTEB publishes as plain files (BEIR's own copy is Parquet).
//!
//! The revision was read from the dataset repository's API. Each
//! digest is the SHA-256 of two separate downloads of the file, which
//! agreed.

use super::{Digest, RemoteFile, Set, Source};
use crate::read::beir;

pub const SET: Set = Set {
    name: "scifact",
    license: "CC BY-SA 4.0, as the BEIR dataset card states it (the MTEB copy's card says unknown)",
    source: Source::Files(&[
        RemoteFile {
            path: beir::CORPUS,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/corpus.jsonl",
            sha256: Digest::Recorded("f0d32db0d156b526d75921ed7a76f2cb912902631c87248c5c97c617bad0b60c"),
        },
        RemoteFile {
            path: beir::QUERIES,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/queries.jsonl",
            sha256: Digest::Recorded("9db7df096f7414435d52bafcbacf814c30cea50eb565c1e8fa6d11440759bba8"),
        },
        RemoteFile {
            path: beir::QRELS,
            url: "https://huggingface.co/datasets/mteb/scifact/resolve/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/qrels/test.tsv",
            sha256: Digest::Recorded("0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"),
        },
    ]),
    span_unit: None,
    read: beir::read_in,
};
