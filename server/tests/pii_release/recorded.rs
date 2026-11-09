//! `--pii-require-recorded-policy` (#1977) over real mTLS: with it on, a
//! values request that names no redaction policy is refused, naming no
//! kind and no value, when the graph's `turbo:pii-edits` recorded one;
//! with it off the request is released as before, and a graph whose
//! edits recorded no policy is released either way.

use tonic::Code;
use turbo_kserve::Server;
use turbo_kserve::pii_policy::PolicyFiles;
use turbo_kserve::proto::result::{PiiValueForm, TypedDocumentResult, pii_value};
use turbo_kserve::release::ReleaseConfig;

use crate::ceiling::{inline, redacted, sent, under, wire};
use crate::certs::client_fingerprint;
use crate::fixture::*;
use crate::{SECOND, forms, reason};

/// An mTLS server granting the client clear for every kind, with the
/// setting `require`.
async fn serving(tag: &str, require: bool) -> (Files, Audit, Server) {
    let files = Files::create(tag);
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:clear:*", client_fingerprint());
    let config = ReleaseConfig { require_recorded_policy: require, ..release(&files, &[grant], at, SECOND) };
    let s = server_with(&files, config, open_policy(), PolicyFiles(Vec::new())).await;
    (files, audit, s)
}

/// The forms of `typed`, which must be both mentions in clear.
fn clear(typed: &TypedDocumentResult) -> Vec<pii_value::Form> {
    let got = forms(typed);
    assert_eq!(got, vec![pii_value::Form::Clear(EMAIL.into()), pii_value::Form::Clear(PHONE.into())]);
    got
}

#[tokio::test]
async fn with_the_setting_a_request_naming_no_policy_is_refused_when_the_graph_recorded_one() {
    let p = wire(None, 1);
    let (_files, audit, s) = serving("recorded-on", true).await;
    let mut c = connect(&s, true).await.unwrap();
    let e = c.inspect_document_graph(under(PiiValueForm::Clear, "op-r", redacted(&p), None)).await.unwrap_err();
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert!(e.message().contains("pii_policy") && e.message().contains("names no policy"), "{e:?}");
    for named in ["email", "EMAIL", "phone", "PHONE", EMAIL, PHONE, "[E]"] {
        assert!(!e.message().contains(named) && !format!("{e:?}").contains(named), "names {named}: {e:?}");
    }
    assert!(audit.calls.lock().unwrap().is_empty(), "nothing released, nothing recorded");
    // Naming the recorded policy is released as without the setting.
    let typed = c.inspect_document_graph(under(PiiValueForm::Clear, "op-n", redacted(&p), inline(&p))).await;
    clear(&typed.unwrap().into_inner());
    assert_eq!(sent(&audit).len(), 1);
    s.stop().await;
}

#[tokio::test]
async fn without_the_setting_a_request_naming_no_policy_is_released_as_before() {
    let p = wire(None, 1);
    let (_files, audit, s) = serving("recorded-off", false).await;
    let mut c = connect(&s, true).await.unwrap();
    let typed = c.inspect_document_graph(under(PiiValueForm::Clear, "op-o", redacted(&p), None)).await.unwrap();
    let got = clear(&typed.into_inner());
    assert!(got.iter().all(|f| reason(f).is_none()), "{got:?}");
    let events = sent(&audit);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].mentions.len(), 2);
    s.stop().await;
}

#[tokio::test]
async fn a_graph_that_recorded_no_policy_is_released_with_the_setting_on_or_off() {
    for (tag, require) in [("unrecorded-on", true), ("unrecorded-off", false)] {
        let (_files, audit, s) = serving(tag, require).await;
        let mut c = connect(&s, true).await.unwrap();
        let req = under(PiiValueForm::Clear, "op-u", graph(), None);
        let typed = c.inspect_document_graph(req).await.unwrap().into_inner();
        clear(&typed);
        assert_eq!(sent(&audit).len(), 1, "{tag}");
        s.stop().await;
    }
}
