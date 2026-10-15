//! ConTEB's SQuAD set: questions over Wikipedia passages cut into chunks,
//! each judged against the one chunk that answers it. It replaces
//! ConTEB's COVID-QA set, which is refused (see
//! [`conteb_covid_qa`](super::conteb_covid_qa)). The validation split is
//! used.
//!
//! - ConTEB's card
//!   (<https://huggingface.co/datasets/illuin-conteb/squad-conteb-eval>,
//!   revision 8f81cbd) states no licence; it says the set stems from
//!   SQuAD.
//! - SQuAD (<https://rajpurkar.github.io/SQuAD-explorer/>; hub copy
//!   `rajpurkar/squad`) is CC BY-SA 4.0, and its passages are Wikipedia
//!   text under the same licence, which ConTEB's chunks reproduce.
//!
//! The data is downloaded when used and never bundled. The tables are
//! published as Parquet, whose digests two downloads and the repository's
//! LFS object ids agree on. The reader reads the files as fetched.

use std::path::Path;

use super::{Digest, RemoteFile, Set, Source, SpanUnit};
use crate::Result;
use crate::read::conteb;
use crate::set::LoadedSet;

pub const DOCUMENTS: &str = "documents/validation-00000-of-00001.parquet";
pub const QUERIES: &str = "queries/validation-00000-of-00001.parquet";

pub const SET: Set = Set {
    name: "conteb-squad",
    license: "CC BY-SA 4.0, as SQuAD states it for its Wikipedia passages, which ConTEB's chunks reproduce; \
              ConTEB's card states none",
    source: Source::Files(&[
        RemoteFile {
            path: DOCUMENTS,
            url: "https://huggingface.co/datasets/illuin-conteb/squad-conteb-eval/resolve/8f81cbdaebe37f78a29f64a60af8111b539ef8bb/documents/validation-00000-of-00001.parquet",
            sha256: Digest::Recorded("5be7bf8e9fa2c1aebf30107f878c4e6104c65dd825df2673bc39eb0c78b341b3"),
        },
        RemoteFile {
            path: QUERIES,
            url: "https://huggingface.co/datasets/illuin-conteb/squad-conteb-eval/resolve/8f81cbdaebe37f78a29f64a60af8111b539ef8bb/queries/validation-00000-of-00001.parquet",
            sha256: Digest::Recorded("4e45dac062415ef5d9cbbf5c7f8c90ced0392a57e36a3482834689b859106ded"),
        },
    ]),
    span_unit: Some(SpanUnit::Chunks),
    read,
};

fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    conteb::read(name, dir, DOCUMENTS, QUERIES)
}
