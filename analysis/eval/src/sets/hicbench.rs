//! HiCBench: questions over documents with hierarchical structure, each
//! with the sentences quoted from the document that answer it.
//!
//! HiChunk's README describes the layout the reader takes, but its data
//! link is empty and no published copy was found, and no licence for the
//! data is stated (the repository's code is MIT). The set is refused
//! until both are recorded here.

use super::{Set, Source, SpanUnit};
use crate::read::hicbench;

pub const SET: Set = Set {
    name: "hicbench",
    license: "",
    source: Source::Unpublished {
        note: "HiChunk's README names dataset/doc/HiCBench/<doc_id>.txt and dataset/qas/HiCBench.jsonl \
               but links to no copy of them",
    },
    span_unit: Some(SpanUnit::QuotedText),
    read: hicbench::read,
};
