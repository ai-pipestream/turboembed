//! GutenQA: questions over 100 Project Gutenberg books cut into chunks by
//! LumberChunker, each with the passage its answer is in.
//!
//! The tables are published as Parquet, whose digests two downloads and
//! the repository's LFS object ids agree on. The reader reads the files
//! as fetched.

use super::{Digest, RemoteFile, Set, Source, SpanUnit};
use crate::read::gutenqa;

pub const SET: Set = Set {
    name: "gutenqa",
    license: "MIT, as the dataset card states it; the books are public domain texts from Project Gutenberg",
    source: Source::Files(&[
        RemoteFile {
            path: "gutenqa_chunks.parquet",
            url: "https://huggingface.co/datasets/LumberChunker/GutenQA/resolve/5f0f6f10d92a819f8efac7c10c0c983033bbe430/gutenqa_chunks.parquet",
            sha256: Digest::Recorded("ee544a9c7e046464c3ff9329b33e848a9d7a29d7936d4cff92ed1b984549397d"),
        },
        RemoteFile {
            path: "questions.parquet",
            url: "https://huggingface.co/datasets/LumberChunker/GutenQA/resolve/5f0f6f10d92a819f8efac7c10c0c983033bbe430/questions.parquet",
            sha256: Digest::Recorded("7d5b9ab95cfd78afa606e77c84d67ff12f0235978d9ab9b03f483af7952ff516"),
        },
    ]),
    span_unit: Some(SpanUnit::QuotedText),
    read: gutenqa::read_in,
};
