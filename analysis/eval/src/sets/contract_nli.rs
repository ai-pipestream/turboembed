//! Pinned test split, supplied from the official archive by the user.
use super::{Set, Source, SpanUnit};
use crate::read::contract_nli;

pub const SET: Set = Set {
    name: "contract-nli",
    license: "CC BY 4.0, Koreeda and Manning, ContractNLI (2021); accompanying TERMS apply",
    source: Source::Manual {
        url: "https://stanfordnlp.github.io/contract-nli/",
        layout: "extract the official test.json into this set directory; the reader checks its pinned SHA-256",
    },
    span_unit: Some(SpanUnit::CodePoints),
    read: contract_nli::read_in,
};
