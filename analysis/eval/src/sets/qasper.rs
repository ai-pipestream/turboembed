//! Qasper v0.3, test split: questions about NLP papers, with the
//! paragraphs that answer them as evidence. The archive was fetched and
//! its digest recorded.

use super::{Digest, RemoteFile, Set, Source, SpanUnit};
use crate::read::qasper;

pub const SET: Set = Set {
    name: "qasper",
    license: "CC BY 4.0, as the Qasper dataset card states it",
    source: Source::Files(&[RemoteFile {
        path: qasper::ARCHIVE,
        url: "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
        sha256: Digest::Recorded("72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"),
    }]),
    span_unit: Some(SpanUnit::QuotedText),
    read: qasper::read,
};
