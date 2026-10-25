//! Schema-extracted records over gRPC with a real client against the
//! server in-process: the `turbo:records` layer crosses the wire and back
//! unchanged in every offset mode, on text whose characters take one to
//! four bytes and one or two UTF-16 units, and a malformed message is
//! refused naming what it found. Nothing is mocked.

mod common;

use turbo_analysis::{Layer, Span, Spans, Tag, TagSet, ViewId};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::RecordsResult;
use turbo_kserve::results::Wire;
use turbo_kserve::results::records::{RecordSet, schema_limits};
use turbo_schema_extract::{Limits, Records};

use common::*;

const TEXT: &str = "Rechnung für Zoë 😀 Müller: Gesamt 42,50 € am 3. März. Bezahlt.";

fn at(needle: &str) -> Span {
    let s = TEXT.find(needle).unwrap();
    Span::new(s as u32, (s + needle.len()) as u32)
}

/// One invoice: a customer and a total (the total corroborated by a
/// number row), a currency inside the total, and a closed choice.
fn records() -> Records {
    let mut spans = Spans::new(ViewId::SOURCE);
    for s in ["Zoë 😀 Müller", "42,50 €", "€"] {
        spans.push(at(s));
    }
    let paths =
        TagSet::new("invoice-fields", &["invoice.customer", "invoice.total", "invoice.currency", "invoice.status"]);
    let r = Records {
        spans,
        record: vec![0, 0, 0],
        path: vec![Tag(0), Tag(1), Tag(2)],
        score: vec![0.91, 0.84, 0.5],
        logit: vec![2.3, 1.66, 0.0],
        support_end: vec![0, 1, 1],
        support_source: vec![2],
        support_row: vec![0],
        record_type: vec![Tag(0)],
        instance: vec![0],
        choice_record: vec![0],
        choice_path: vec![Tag(3)],
        choice: vec![Tag(0)],
        choice_score: vec![0.97],
        choice_logit: vec![3.5],
        types: TagSet::new("record-types", &["invoice"]),
        paths,
        choices: TagSet::new("choices", &["paid", "open"]),
    };
    r.check().unwrap();
    r
}

fn set(offsets: OffsetMode) -> RecordSet {
    RecordSet { model: model_ref(), text: TEXT.into(), view: ViewId::SOURCE, offsets, records: records() }
}

fn refused(msg: &RecordsResult, needle: &str) {
    match RecordSet::from_msg(msg) {
        Ok(_) => panic!("accepted; wanted a refusal naming {needle:?}"),
        Err(e) => assert!(e.message.contains(needle), "{needle:?} not in {:?}", e.message),
    }
}

#[tokio::test]
async fn records_cross_the_wire_unchanged_in_every_offset_mode() {
    let (_s, mut c) = serve().await;
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let x = set(mode);
        let msg = x.to_msg();
        let back = c.round_trip_records(msg.clone()).await.unwrap().into_inner();
        assert_eq!(back, msg, "{mode:?}: the answer is the request");
        let d = RecordSet::from_msg(&back).unwrap();
        let (a, b) = (&d.records, &x.records);
        assert_eq!(a.spans.columns(), b.spans.columns(), "{mode:?}: byte spans exact");
        assert_eq!((&a.record, &a.path, &a.score, &a.logit), (&b.record, &b.path, &b.score, &b.logit));
        assert_eq!(
            (&a.support_end, &a.support_source, &a.support_row),
            (&b.support_end, &b.support_source, &b.support_row)
        );
        assert_eq!((&a.record_type, &a.instance), (&b.record_type, &b.instance));
        assert_eq!((&a.choice_record, &a.choice_path, &a.choice), (&b.choice_record, &b.choice_path, &b.choice));
        assert_eq!((&a.choice_score, &a.choice_logit), (&b.choice_score, &b.choice_logit));
        assert_eq!(a.paths.label(Tag(1)), "invoice.total");
        assert_eq!((a.types.name.as_str(), a.choices.label(Tag(0))), ("record-types", "paid"));
        assert_eq!(d.model, x.model);
        // The wire offsets are counted in the mode's own units.
        let customer = TEXT.find("Zoë").unwrap();
        let before = &TEXT[..customer];
        let value = "Zoë 😀 Müller";
        let (s, len) = match mode {
            OffsetMode::Utf8 => (before.len(), value.len()),
            OffsetMode::Utf16 => (before.encode_utf16().count(), value.encode_utf16().count()),
            OffsetMode::CodePoint => (before.chars().count(), value.chars().count()),
        };
        assert_eq!((msg.value_frag_start[0], msg.value_frag_end[0]), (s as u32, (s + len) as u32), "{mode:?}");
        let got: Vec<&str> = (0..a.len() as u32)
            .map(|i| a.spans.get(turbo_analysis::Ix::new(i)))
            .map(|s| &TEXT[s.start as usize..s.end as usize])
            .collect();
        assert_eq!(got, ["Zoë 😀 Müller", "42,50 €", "€"]);
    }
}

#[test]
fn malformed_records_are_refused_naming_what_is_wrong() {
    let good = set(OffsetMode::Utf16).to_msg();
    // The emoji is two UTF-16 units: an end between them splits it.
    let mut m = good.clone();
    let emoji = TEXT[..TEXT.find('😀').unwrap()].encode_utf16().count() as u32;
    m.value_frag_end[0] = emoji + 1;
    refused(&m, "record value");
    let mut m = good.clone();
    m.paths[2] = "receipt.currency".into();
    refused(&m, "not a field");
    let mut m = good.clone();
    m.paths[1] = "invoice.customer".into();
    refused(&m, "listed twice");
    let mut m = good.clone();
    m.value_path[0] = 9;
    refused(&m, "value_path");
    let mut m = good.clone();
    m.record_type.push(0);
    m.record_instance.push(1);
    refused(&m, "no value");
    let mut m = good.clone();
    m.value_score[0] = 1.5;
    refused(&m, "score");
    let mut m = good.clone();
    m.choice_logit[0] = f64::NAN;
    refused(&m, "logit");
    let mut m = good.clone();
    m.support_source[0] = 9;
    refused(&m, "support source");
    let mut m = good.clone();
    m.support_source[0] = 300;
    refused(&m, "support_source");
    let mut m = good.clone();
    m.value_logit.pop();
    refused(&m, "value_logit");
    let mut m = good.clone();
    m.model = None;
    refused(&m, "model identity");
    let mut m = good.clone();
    m.offsets = 7;
    refused(&m, "offset mode");
    let mut m = good.clone();
    m.value_frag_end[1] = 4000;
    refused(&m, "record value");
    assert!(RecordSet::from_msg(&good).is_ok());
}

#[test]
fn schema_limits_are_the_crates_own() {
    let l = Limits::default();
    let w = schema_limits(&l);
    assert_eq!(w.max_entity_types as usize, l.max_entity_types);
    assert_eq!(w.max_classification_tasks as usize, l.max_classification_tasks);
    assert_eq!(w.max_labels_per_task as usize, l.max_labels_per_task);
    assert_eq!(w.max_structures as usize, l.max_structures);
    assert_eq!(w.max_fields_per_structure as usize, l.max_fields_per_structure);
    assert_eq!(w.max_text_bytes as usize, l.max_text_bytes);
    assert_eq!(w.max_prompt_subwords as usize, l.max_prompt_subwords);
    assert_eq!(w.max_words as usize, l.max_words);
    assert_eq!(schema_limits(&Limits { max_words: usize::MAX, ..l }).max_words, u32::MAX);
}
