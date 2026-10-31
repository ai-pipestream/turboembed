//! Pinned test split, supplied from the official archive by the user.
use super::{Set, Source, SpanUnit};
use crate::read::cuad;

pub const SET: Set = Set {
    name: "cuad",
    license: "CC BY 4.0, The Atticus Project CUAD v1",
    source: Source::Manual {
        url: "https://www.atticusprojectai.org/cuad/",
        layout: "extract the official test.json into this set directory; the reader checks its pinned SHA-256",
    },
    span_unit: Some(SpanUnit::CodePoints),
    read: cuad::read_in,
};
