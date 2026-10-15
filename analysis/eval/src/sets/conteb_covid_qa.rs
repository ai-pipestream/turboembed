//! ConTEB's COVID-QA set: questions over research papers cut into
//! chunks, each judged against the one chunk that answers it. The tables
//! are published as Parquet, which the reader reads as fetched.
//!
//! The set is refused: no licence covering its text was found (checked
//! 2026-10-15).
//!
//! - ConTEB's card (<https://huggingface.co/datasets/illuin-conteb/covid-qa>,
//!   revision df93738) states no licence and says all rights are reserved
//!   to the original document authors. None of the illuin-conteb datasets
//!   on the hub carries a licence. ConTEB's code
//!   (<https://github.com/illuin-tech/conteb>) is MIT, which covers the
//!   code.
//! - Its source, COVID-QA by deepset
//!   (<https://github.com/deepset-ai/COVID-QA>; paper
//!   <https://aclanthology.org/2020.nlpcovid19-acl.18>), is in a
//!   repository under Apache 2.0, and the hub copy
//!   `deepset/covid_qa_deepset` states Apache 2.0. Its contexts are the
//!   full text of 147 articles taken from CORD-19; neither the paper nor
//!   the repository says which licence each article has.
//! - CORD-19 (<https://arxiv.org/abs/2004.10706>) passes each article on
//!   under the article's own licence, recorded in its metadata, and says
//!   some of those licences restrict redistribution or commercial use.
//!
//! So Apache 2.0 covers deepset's annotations, but the article text that
//! ConTEB chunks stays under its authors' and publishers' terms, as
//! ConTEB's card says. docs/analysis.md ("The sets") proposes a
//! replacement.

use std::path::Path;

use super::{Digest, RemoteFile, Set, Source, SpanUnit};
use crate::Result;
use crate::read::conteb;
use crate::set::LoadedSet;

const DOCUMENTS: &str = "documents/train-00000-of-00001.parquet";
const QUERIES: &str = "queries/train-00000-of-00001.parquet";

pub const SET: Set = Set {
    name: "conteb-covid-qa",
    license: "",
    source: Source::Files(&[
        RemoteFile {
            path: DOCUMENTS,
            url: "https://huggingface.co/datasets/illuin-conteb/covid-qa/resolve/df9373821efad895edf07b50a31ecd652b679b93/documents/train-00000-of-00001.parquet",
            sha256: Digest::NotRecorded,
        },
        RemoteFile {
            path: QUERIES,
            url: "https://huggingface.co/datasets/illuin-conteb/covid-qa/resolve/df9373821efad895edf07b50a31ecd652b679b93/queries/train-00000-of-00001.parquet",
            sha256: Digest::NotRecorded,
        },
    ]),
    span_unit: Some(SpanUnit::Chunks),
    read,
};

fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    conteb::read(name, dir, DOCUMENTS, QUERIES)
}
