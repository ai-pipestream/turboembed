//! A redaction policy's per-kind release ceiling (#1943, #1977) over real
//! mTLS: a values request that names the policy the graph's
//! `turbo:pii-edits` were written with releases at most the forms the
//! kind's ceiling lists, withholds the rest as not allowed by policy, and
//! is refused, naming no kind and no value, when the policy is not the
//! graph's or when the server requires the recorded policy and the request
//! names none.

use std::sync::Arc;

use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use tonic::Code;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, View};
use turbo_kserve::Server;
use turbo_kserve::pii_policy::{NoRegistry, PolicyFiles, REDACT_STEP, RedactStep, RedactionPolicy};
use turbo_kserve::proto::audit::{PiiAccessEvent, PiiAccessForm, RecordPiiAccessRequest};
use turbo_kserve::proto::result::{
    self as pb, InspectDocumentGraphRequest, PiiReleasePolicy, PiiValueForm, PiiWithheldReason, pii_release_policy,
    pii_value,
};
use turbo_kserve::results::document::codecs;
use turbo_pii::rewrite::REDACTED;
use turbo_pii::{Pii, PiiEdits};

use crate::certs::client_fingerprint;
use crate::fixture::*;
use crate::policy::{builtin, entry, rule};
use crate::{SECOND, ask, forms, reason};

/// A policy whose email rule has `ceiling` and whose phone kind has no
/// rule, at `revision`.
fn wire(ceiling: Option<&[PiiValueForm]>, revision: u64) -> pb::RedactionPolicy {
    let mut email = rule(
        "email",
        builtin(pb::PiiKindId::Email),
        vec![entry(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() }))],
    );
    email.release_ceiling = ceiling.map(|f| pb::PiiReleaseCeiling { forms: f.iter().map(|&f| f as i32).collect() });
    let mut p = crate::policy::policy(vec![email]);
    p.revision = revision;
    p
}

/// `graph()` with the redacted view and edits `policy` writes.
fn redacted(policy: &pb::RedactionPolicy) -> Vec<u8> {
    let policy = Arc::new(RedactionPolicy::from_proto(policy, &NoRegistry).unwrap());
    let mut doc: Document = import(&DocumentGraph::from_bytes(&graph()).unwrap(), codecs()).unwrap();
    let (view, mut edits) = {
        let src = doc.get_key(&Key::<View>::DEFAULT).unwrap();
        RedactStep::new(policy).rewrite(src, doc.get::<Pii>().unwrap()).unwrap()
    };
    doc.add(&REDACTED, view, Provenance::by(REDACT_STEP)).unwrap();
    edits.bind_pending_view(doc.get_key(&REDACTED).unwrap().id());
    doc.add(&Key::<PiiEdits>::named(REDACTED.name), edits, Provenance::by(REDACT_STEP)).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

/// `graph()` with a `turbo:pii-edits` layer that records no policy digest.
fn unrecorded() -> Vec<u8> {
    let policy = Arc::new(RedactionPolicy::from_proto(&crate::policy::policy(vec![]), &NoRegistry).unwrap());
    let mut doc: Document = import(&DocumentGraph::from_bytes(&graph()).unwrap(), codecs()).unwrap();
    let (view, mut edits) = {
        let src = doc.get_key(&Key::<View>::DEFAULT).unwrap();
        RedactStep::new(policy).rewrite(src, doc.get::<Pii>().unwrap()).unwrap()
    };
    edits.policy_digest = None;
    doc.add(&REDACTED, view, Provenance::by(REDACT_STEP)).unwrap();
    edits.bind_pending_view(doc.get_key(&REDACTED).unwrap().id());
    doc.add(&Key::<PiiEdits>::named(REDACTED.name), edits, Provenance::by(REDACT_STEP)).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn inline(p: &pb::RedactionPolicy) -> Option<PiiReleasePolicy> {
    Some(PiiReleasePolicy { policy: Some(pii_release_policy::Policy::Inline(p.encode_to_vec())) })
}

fn named(name: &str) -> Option<PiiReleasePolicy> {
    Some(PiiReleasePolicy { policy: Some(pii_release_policy::Policy::Name(name.into())) })
}

/// A request for `form` from `graph`, released under `policy`.
fn under(
    form: PiiValueForm,
    key: &str,
    graph: Vec<u8>,
    policy: Option<PiiReleasePolicy>,
) -> InspectDocumentGraphRequest {
    InspectDocumentGraphRequest { graph, pii_policy: policy, ..ask(form, key) }
}

fn sent(audit: &Audit) -> Vec<PiiAccessEvent> {
    let calls = audit.calls.lock().unwrap().clone();
    calls.iter().map(|c| RecordPiiAccessRequest::decode(c.as_slice()).unwrap().event.unwrap()).collect()
}

/// An mTLS server granting the client token and clear for every kind and
/// serving `served` by name.
async fn granted(tag: &str, served: &[pb::RedactionPolicy]) -> (Files, Audit, Server) {
    granted_with(tag, served, false).await
}

async fn granted_with(
    tag: &str,
    served: &[pb::RedactionPolicy],
    require_recorded_policy: bool,
) -> (Files, Audit, Server) {
    let files = Files::create(tag);
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token+clear:*", client_fingerprint());
    let mut config = release(&files, &[grant], at, SECOND);
    config.require_recorded_policy = require_recorded_policy;
    let served = PolicyFiles(served.iter().map(Message::encode_to_vec).collect());
    let s = server_with(&files, config, open_policy(), served).await;
    (files, audit, s)
}

#[tokio::test]
async fn a_token_ceiling_with_a_clear_grant_releases_a_token_and_no_clear_text() {
    let p = wire(Some(&[PiiValueForm::Token]), 1);
    let (_files, audit, s) = granted("ceiling-token", std::slice::from_ref(&p)).await;
    let mut c = connect(&s, true).await.unwrap();
    for (tag, sel) in [("inline", inline(&p)), ("named", named("test-policy"))] {
        // Clear is granted for the email kind, and the ceiling withholds it.
        let req = under(PiiValueForm::Clear, &format!("op-clear-{tag}"), redacted(&p), sel.clone());
        let typed = c.inspect_document_graph(req).await.unwrap().into_inner();
        let got = forms(&typed);
        assert_eq!(reason(&got[0]), Some(PiiWithheldReason::NotAllowedByPolicy), "{tag}: {got:?}");
        assert_eq!(got[1], pii_value::Form::Clear(PHONE.into()), "{tag}: a kind with no ceiling is released");
        assert!(!holds(&typed.encode_to_vec(), EMAIL.as_bytes()), "{tag}: no clear email");
        // A token is within the ceiling.
        let req = under(PiiValueForm::Token, &format!("op-token-{tag}"), redacted(&p), sel);
        let typed = c.inspect_document_graph(req).await.unwrap().into_inner();
        let got = forms(&typed);
        assert!(matches!(&got[0], pii_value::Form::Token(t) if t.token.len() == 32), "{tag}: {got:?}");
        assert!(!holds(&typed.encode_to_vec(), EMAIL.as_bytes()), "{tag}: a token carries no clear value");
    }
    let events = sent(&audit);
    assert_eq!(events.len(), 4);
    let email = pb::PiiKindId::Email as u32;
    for e in &events {
        let grant = e.effective.iter().find(|g| g.builtin_kind == email);
        let mention = e.mentions.iter().find(|m| m.builtin_kind == email);
        match mention {
            Some(m) => {
                assert_eq!(m.form, PiiAccessForm::Token as i32, "only a token of the email kind is released");
                assert_eq!(grant.unwrap().forms, vec![PiiAccessForm::Token as i32], "the ceiling is in effect");
            }
            None => assert!(grant.is_none(), "an unreleased kind has no effective grant"),
        }
    }
    s.stop().await;
}

#[tokio::test]
async fn an_empty_ceiling_withholds_that_kind_and_releases_the_others() {
    let p = wire(Some(&[]), 1);
    let (_files, audit, s) = granted("ceiling-empty", &[]).await;
    let mut c = connect(&s, true).await.unwrap();
    for form in [PiiValueForm::Clear, PiiValueForm::Token] {
        let typed = c.inspect_document_graph(under(form, "op-empty", redacted(&p), inline(&p))).await.unwrap();
        let got = forms(&typed.into_inner());
        assert_eq!(reason(&got[0]), Some(PiiWithheldReason::NotAllowedByPolicy), "{form:?}: {got:?}");
        assert_eq!(reason(&got[1]), None, "{form:?}: the phone kind is released: {got:?}");
    }
    let events = sent(&audit);
    assert_eq!(events.len(), 2);
    for e in &events {
        assert!(e.mentions.iter().all(|m| m.builtin_kind == pb::PiiKindId::Phone as u32), "{e:?}");
        assert!(e.effective.iter().all(|g| g.builtin_kind == pb::PiiKindId::Phone as u32), "{e:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn an_unset_ceiling_behaves_as_today() {
    let p = wire(None, 1);
    let (_files, audit, s) = granted("ceiling-unset", &[]).await;
    let mut c = connect(&s, true).await.unwrap();
    let graph = redacted(&p);
    let without = c.inspect_document_graph(under(PiiValueForm::Clear, "op-a", graph.clone(), None)).await.unwrap();
    let with = c.inspect_document_graph(under(PiiValueForm::Clear, "op-b", graph, inline(&p))).await.unwrap();
    assert_eq!(forms(&without.into_inner()), forms(&with.into_inner()));
    let events = sent(&audit);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].mentions.len(), 2);
    assert_eq!(events[0].effective, events[1].effective);
    assert_ne!(events[0].policy_digest, events[1].policy_digest, "the named policy is part of the event's policy");
    s.stop().await;
}

#[tokio::test]
async fn a_policy_that_is_not_the_graphs_is_refused_naming_no_kind_and_no_value() {
    let (r1, r2) = (wire(Some(&[PiiValueForm::Token]), 1), wire(Some(&[PiiValueForm::Token]), 2));
    let (_files, audit, s) = granted("ceiling-mismatch", std::slice::from_ref(&r2)).await;
    let mut c = connect(&s, true).await.unwrap();
    let refusals = [
        ("another revision", under(PiiValueForm::Clear, "op-m", redacted(&r1), inline(&r2))),
        ("another named revision", under(PiiValueForm::Clear, "op-m", redacted(&r1), named("test-policy"))),
        ("a graph with no edits", under(PiiValueForm::Clear, "op-m", graph(), inline(&r1))),
    ];
    for (tag, req) in refusals {
        let e = c.inspect_document_graph(req).await.unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument, "{tag}: {e:?}");
        assert!(e.message().contains("pii_policy"), "{tag}: {e:?}");
        for named in ["email", "EMAIL", "phone", "PHONE", EMAIL, PHONE, "[E]"] {
            assert!(!e.message().contains(named) && !format!("{e:?}").contains(named), "{tag} names {named}: {e:?}");
        }
    }
    // A selection that does not resolve is refused before the graph is read.
    let unread = |sel| under(PiiValueForm::Clear, "op-m", b"not a graph".to_vec(), sel);
    let not_a_policy = Some(PiiReleasePolicy { policy: Some(pii_release_policy::Policy::Inline(vec![0xff])) });
    for (tag, sel) in
        [("unknown name", named("nope")), ("bad bytes", not_a_policy), ("unset", Some(Default::default()))]
    {
        let e = c.inspect_document_graph(unread(sel)).await.unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument, "{tag}: {e:?}");
        assert!(e.message().contains("pii_policy") && !e.message().contains("nope"), "{tag}: {e:?}");
    }
    assert!(audit.calls.lock().unwrap().is_empty(), "nothing released, nothing recorded");
    s.stop().await;
}

#[tokio::test]
async fn requiring_the_recorded_policy_refuses_an_unnamed_policy_and_leaves_unrecorded_graphs_alone() {
    let p = wire(Some(&[PiiValueForm::Token]), 1);
    let (_files_on, audit_on, s_on) = granted_with("require-on", std::slice::from_ref(&p), true).await;
    let mut c_on = connect(&s_on, true).await.unwrap();
    // Setting on: a graph whose turbo:pii-edits recorded a digest refuses a
    // values request naming no policy, naming no kind and no value.
    for form in [PiiValueForm::Clear, PiiValueForm::Token] {
        let e = c_on.inspect_document_graph(under(form, "op-req", redacted(&p), None)).await.unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument, "{form:?}: {e:?}");
        assert!(e.message().contains("pii_policy"), "{form:?}: {e:?}");
        for named in ["email", "EMAIL", "phone", "PHONE", EMAIL, PHONE, "[E]"] {
            assert!(!e.message().contains(named) && !format!("{e:?}").contains(named), "{form:?} names {named}: {e:?}");
        }
    }
    assert!(audit_on.calls.lock().unwrap().is_empty(), "nothing released, nothing recorded");
    // Naming the recorded policy still releases under its ceiling.
    let typed = c_on
        .inspect_document_graph(under(PiiValueForm::Clear, "op-named", redacted(&p), named("test-policy")))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(reason(&forms(&typed)[0]), Some(PiiWithheldReason::NotAllowedByPolicy));
    assert_eq!(forms(&typed)[1], pii_value::Form::Clear(PHONE.into()));

    // Setting off (default): a values request naming no policy releases as today,
    // even when the graph's edits recorded a policy digest.
    let (_files_off, audit_off, s_off) = granted_with("require-off", std::slice::from_ref(&p), false).await;
    let mut c_off = connect(&s_off, true).await.unwrap();
    let typed_off = c_off
        .inspect_document_graph(under(PiiValueForm::Clear, "op-off", redacted(&p), None))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        forms(&typed_off),
        vec![pii_value::Form::Clear(EMAIL.into()), pii_value::Form::Clear(PHONE.into())],
        "with the setting off, both mentions are released under the grant"
    );
    assert_eq!(sent(&audit_off).len(), 1);

    // A graph with no recorded digest (either no turbo:pii-edits layer or an
    // edits layer with policy_digest == None) is unaffected either way.
    for (tag, g) in [("no-edits", graph()), ("unrecorded-edits", unrecorded())] {
        let on = c_on
            .inspect_document_graph(under(PiiValueForm::Clear, &format!("op-{tag}"), g.clone(), None))
            .await
            .unwrap()
            .into_inner();
        let off = c_off
            .inspect_document_graph(under(PiiValueForm::Clear, &format!("op-{tag}"), g, None))
            .await
            .unwrap()
            .into_inner();
        let expected = vec![pii_value::Form::Clear(EMAIL.into()), pii_value::Form::Clear(PHONE.into())];
        assert_eq!(forms(&on), expected, "{tag} on");
        assert_eq!(forms(&off), expected, "{tag} off");
    }
    assert_eq!(sent(&audit_on).len(), 3, "the named-policy release and the two unrecorded graphs");
    assert_eq!(sent(&audit_off).len(), 3);
    s_on.stop().await;
    s_off.stop().await;
}
