use super::*;

fn m(start: u32, end: u32, kind: PiiKind, normalized: &str) -> PiiMention<'_> {
    PiiMention { span: Span::new(start, end), kind, confidence: PiiConfidence::Format, normalized }
}

#[test]
fn kinds_have_the_java_names_and_ranks() {
    assert_eq!(PiiKind::ALL.len(), 20);
    for (rank, kind) in PiiKind::ALL.into_iter().enumerate() {
        assert_eq!(kind.rank() as usize, rank);
        assert_eq!(PiiKind::parse(kind.name()), Some(kind));
        assert_eq!(kind.to_string(), kind.name());
    }
    assert_eq!(PiiKind::DeSteuerId.name(), "de-steuer-id");
    assert!(PiiKind::Imei < PiiKind::Card);
    assert_eq!(PiiKind::parse("EMAIL"), None);
}

#[test]
fn kind_sets() {
    let s = PiiKinds::of(&[PiiKind::Phone, PiiKind::Email]);
    assert!(s.contains(PiiKind::Email) && !s.contains(PiiKind::Iban));
    assert_eq!(s.iter().collect::<Vec<_>>(), [PiiKind::Email, PiiKind::Phone]);
    assert!(s.is_subset(s.union(PiiKinds::of(&[PiiKind::Iban]))));
    assert!(!s.is_subset(PiiKinds::of(&[PiiKind::Email])));
    assert!(PiiKinds::EMPTY.is_empty());
}

#[test]
fn mentions_round_trip_and_may_overlap() {
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 15, PiiKind::Imei, "490154203237518"));
    p.push(PiiMention { confidence: PiiConfidence::Checksum, ..m(0, 15, PiiKind::Card, "490154203237518") });
    p.push(m(5, 20, PiiKind::Phone, "1234567890"));
    assert_eq!(p.check(), Ok(()));
    assert_eq!(p.len(), 3);
    assert_eq!(p.get(Ix::new(1)).confidence, PiiConfidence::Checksum);
    assert_eq!(p.get(Ix::new(2)).normalized, "1234567890");
    assert_eq!(Pii::ID, "turbo:pii");
    assert_eq!(PiiConfidence::LabelledChecksum.name(), "labelled-checksum");
}

#[test]
fn check_wants_the_java_order() {
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Email, "a"));
    p.push(m(0, 9, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Phone, "a"));
    p.push(m(0, 5, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
}

#[test]
fn check_refuses_duplicates_empty_values_and_short_columns() {
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Email, "a"));
    p.push(m(0, 5, PiiKind::Email, "b"));
    assert_eq!(p.check(), Ok(()));
    p.push(m(0, 5, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("repeats"));
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Email, ""));
    assert!(p.check().unwrap_err().contains("no normalized"));
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(3, 3, PiiKind::Email, "x"));
    assert!(p.check().unwrap_err().contains("empty"));
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 3, PiiKind::Email, "x"));
    p.kind.pop();
    assert_eq!(p.check().unwrap_err(), "every column as long as the spans");
}
