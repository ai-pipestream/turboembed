//! A document run through `turbo-pii:pii` and `turbo-pii:redact` with a
//! policy message, for the redact step tests (issue #1882 part B).
#![allow(dead_code)]

use std::sync::Arc;

use turbo_analysis::{Document, Key, View};
use turbo_kserve::pii_policy::{PolicyResolver, Redaction, RedactionPolicy};
use turbo_kserve::progressive::compile_analyzer_with_redaction;
use turbo_kserve::proto::result as pb;
use turbo_pii::PiiEdits;
use turbo_pii::rewrite::REDACTED;

/// The planted values: no edit, decision, receipt, error or log may hold
/// them.
pub const EMAIL: &str = "jane@example.com";
pub const CARD: &str = "4111 1111 1111 1111";
pub const TEXT: &str = "Mail jane@example.com or pay with card 4111 1111 1111 1111 today.";

pub const STEPS: &[&str] = &["turbo-pii:pii", "turbo-pii:redact"];

pub fn convert(msg: &pb::RedactionPolicy, resolver: &dyn PolicyResolver) -> Arc<RedactionPolicy> {
    Arc::new(RedactionPolicy::from_proto(msg, resolver).expect("the policy converts"))
}

/// `TEXT` analyzed with `policy`, as a dry run when `dry_run`.
pub fn run_with(text: &str, policy: Arc<RedactionPolicy>, dry_run: bool) -> Document {
    let redaction = Redaction { policy, dry_run };
    let analyzer = compile_analyzer_with_redaction(&Document::new(""), STEPS, &redaction).expect("the pipeline builds");
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.expect("the pipeline runs");
    }
    docs.pop().unwrap()
}

pub fn run(policy: Arc<RedactionPolicy>) -> Document {
    run_with(TEXT, policy, false)
}

pub fn redacted(doc: &Document) -> (&View, &PiiEdits) {
    let view = doc.get_key(&REDACTED).expect("the redacted view");
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).expect("its edits");
    (view, edits)
}
