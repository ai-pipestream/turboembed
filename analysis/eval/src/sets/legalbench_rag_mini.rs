//! LegalBench-RAG mini: legal questions over contracts and privacy
//! policies (CUAD, MAUD, ContractNLI, PrivacyQA), 194 per benchmark, with
//! character spans as judgements.
//!
//! The data is published as a shared folder, whose download is an
//! archive made on request, so it has no stable digest and no file by
//! file URL. The folder could not be reached from where this was
//! written; the reader follows the layout and types in the benchmark's
//! repository.

use super::{Set, Source, SpanUnit};
use crate::read::legalbench_rag;

pub const SET: Set = Set {
    name: "legalbench-rag-mini",
    license: "MIT for the benchmark (its repository's LICENSE); the documents come from CUAD, MAUD, \
              ContractNLI and PrivacyQA, whose terms its README asks users to agree to",
    source: Source::Manual {
        url: "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0",
        layout: "the folder's benchmarks/ and corpus/ directories, as they are, so that \
                 benchmarks/<name>.json and corpus/<dataset>/<file> are under the set's directory",
    },
    span_unit: Some(SpanUnit::CodePoints),
    read: legalbench_rag::read,
};
