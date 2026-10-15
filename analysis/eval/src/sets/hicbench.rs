//! HiCBench: questions over documents with hierarchical structure, each
//! with the sentences quoted from the document that answer it.
//!
//! HiChunk's README describes the layout the reader takes. The set is
//! refused: no copy can be fetched and no licence for the data is stated
//! (checked 2026-10-15).
//!
//! - The README of <https://github.com/TencentCloudADP/hichunk> (commit
//!   8d17d18) has a data badge that links nowhere; the only data it links
//!   is an example LongBench set on Google Drive, not HiCBench. The repository is MIT
//!   except a CC BY-NC 4.0 Contriever component, and says nothing of the
//!   benchmark's data.
//! - The paper (<https://arxiv.org/abs/2509.11552>, ACL 2026) names
//!   <https://huggingface.co/datasets/Youtu-RAG/HiCBench> as its data,
//!   but that page could not be opened, and the hub lists no public
//!   datasets for Youtu-RAG and none named HiCBench. The paper states no licence.
//! - HiCBench's documents are chosen from OHR-Bench
//!   (<https://huggingface.co/datasets/opendatalab/OHR-Bench>), whose card
//!   states CC BY 4.0. The questions and evidence are HiCBench's own, so
//!   that licence does not settle the set's.

use super::{Set, Source, SpanUnit};
use crate::read::hicbench;

pub const SET: Set = Set {
    name: "hicbench",
    license: "",
    source: Source::Unpublished {
        note: "HiChunk's README names dataset/doc/HiCBench/<doc_id>.txt and dataset/qas/HiCBench.jsonl \
               and the paper names huggingface.co/datasets/Youtu-RAG/HiCBench, which the hub does not list",
    },
    span_unit: Some(SpanUnit::QuotedText),
    read: hicbench::read,
};
