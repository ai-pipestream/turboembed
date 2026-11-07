use super::*;

fn m(start: u32, end: u32, kind: PiiKind, normalized: &str) -> PiiMention<'_> {
    PiiMention { span: Span::new(start, end), kind, confidence: PiiConfidence::Format, normalized }
}

#[test]
fn kinds_have_stable_ids_names_and_priorities() {
    assert_eq!(PiiKind::TABLE.len(), 30);
    for (i, row) in PiiKind::TABLE.iter().enumerate() {
        assert_eq!(usize::from(row.id), i + 1, "ids are 1.. in table order");
        assert_eq!(row.kind.builtin_id(), Some(row.id));
        assert_eq!(row.kind.code(), u32::from(row.id));
        assert_eq!(row.kind.builtin_name(), Some(row.name));
        assert_eq!(row.kind.priority(), row.priority);
        assert_eq!(PiiKind::parse(row.name), Some(row.kind));
        assert_eq!(PiiKind::from_id(row.id), Some(row.kind));
        assert_eq!(row.kind.to_string(), row.name);
    }
    // Pinned: an id never changes once given, so these hold forever.
    let pinned = [
        (PiiKind::Jwt, 1),
        (PiiKind::Email, 5),
        (PiiKind::Card, 8),
        (PiiKind::UsSsn, 14),
        (PiiKind::DeSteuerId, 17),
        (PiiKind::EsDni, 18),
        (PiiKind::BrCnpj, 26),
        (PiiKind::Bic, 27),
        (PiiKind::Phone, 30),
    ];
    for (kind, id) in pinned {
        assert_eq!(kind.builtin_id(), Some(id), "{kind}");
        assert_eq!(kind.bit(), 1u128 << id, "{kind}");
    }
    assert_eq!(PiiKind::from_id(0), None);
    assert_eq!(PiiKind::from_id(31), None);
    assert_eq!(PiiKind::parse("EMAIL"), None);
    assert!(PiiKind::Imei.priority() < PiiKind::Card.priority());
}

#[test]
fn confidence_ids_are_the_c_codes() {
    assert_eq!(PiiConfidence::Format.id(), 1);
    assert_eq!(PiiConfidence::Checksum.id(), 2);
    assert_eq!(PiiConfidence::LabelledChecksum.id(), 3);
    assert_eq!(PiiConfidence::Labelled.id(), 4);
    for c in PiiConfidence::ALL {
        assert_eq!(PiiConfidence::from_id(u32::from(c.id())), Some(c));
        assert_eq!(PiiConfidence::parse(c.name()), Some(c));
    }
    assert_eq!(PiiConfidence::from_id(0), None);
    assert_eq!(PiiConfidence::from_id(5), None);
    assert_eq!(PiiConfidence::Format.name(), "format");
    assert_eq!(PiiConfidence::Labelled.name(), "labelled");
    assert_eq!(PiiConfidence::Checksum.name(), "checksum");
    assert_eq!(PiiConfidence::LabelledChecksum.name(), "labelled-checksum");
    assert_eq!(PiiConfidence::Format.with_label(), PiiConfidence::Labelled);
    assert_eq!(PiiConfidence::Labelled.with_label(), PiiConfidence::Labelled);
    assert_eq!(PiiConfidence::Checksum.with_label(), PiiConfidence::LabelledChecksum);
    assert_eq!(PiiConfidence::LabelledChecksum.with_label(), PiiConfidence::LabelledChecksum);
}

#[test]
fn kind_sets_key_on_ids() {
    let s = PiiKinds::of(&[PiiKind::Phone, PiiKind::Email]);
    assert!(s.contains(PiiKind::Email) && !s.contains(PiiKind::Iban));
    assert_eq!(s.iter().collect::<Vec<_>>(), [PiiKind::Email, PiiKind::Phone]);
    assert_eq!(s.bits(), 1u128 << 5 | 1u128 << 30);
    assert!(s.is_subset(s.union(PiiKinds::of(&[PiiKind::Iban]))));
    assert!(!s.is_subset(PiiKinds::of(&[PiiKind::Email])));
    assert!(PiiKinds::EMPTY.is_empty());
    let custom = CustomKinds::from_names(["ticket"]).unwrap().find("ticket").unwrap();
    assert_eq!(PiiKinds::of(&[custom]), PiiKinds::EMPTY, "custom kinds are not set members");
}

#[test]
fn custom_kind_tables_validate_at_construction() {
    let mut t = CustomKinds::new();
    let a = t.add("account-number").unwrap();
    assert_eq!(a, PiiKind::Custom(CustomKindId::new(0)));
    assert_eq!(t.add("account-number"), Ok(a), "a repeated name is the same kind");
    assert_eq!(t.add(""), Err(KindError::EmptyName));
    assert_eq!(t.add("email"), Err(KindError::Reserved(PiiKind::Email)));
    let long = "q".repeat(CustomKinds::MAX_NAME_BYTES + 1);
    assert_eq!(t.add(&long), Err(KindError::NameTooLong(65)));
    assert!(!KindError::NameTooLong(65).to_string().contains('q'), "the error carries no name");
    assert!(t.add(&"x".repeat(CustomKinds::MAX_NAME_BYTES)).is_ok());
    assert_eq!(CustomKinds::from_names(["a", "a"]), Err(KindError::Duplicate(1)));
    let mut full = CustomKinds::new();
    for i in 0..CustomKinds::MAX {
        full.add(&format!("k{i}")).unwrap();
    }
    assert_eq!(full.add("one-more"), Err(KindError::Full));
    assert_eq!(full.add("k7").unwrap(), PiiKind::Custom(CustomKindId::new(7)));

    assert_eq!(t.name(a), Some("account-number"));
    assert_eq!(t.name(PiiKind::Email), Some("email"));
    assert_eq!(t.resolve("email"), Some(PiiKind::Email));
    assert_eq!(t.resolve("account-number"), Some(a));
    assert_eq!(t.from_code(a.code()), Ok(a));
    assert_eq!(t.from_code(5), Ok(PiiKind::Email));
    assert_eq!(t.from_code(0), Err(KindError::UnknownCode(0)));
    assert_eq!(t.from_code(31), Err(KindError::UnknownCode(31)));
    assert_eq!(t.from_code(CUSTOM_CODE_BASE + 9), Err(KindError::UnknownCode(CUSTOM_CODE_BASE + 9)));
    let other = CustomKinds::from_names(["other", "account-number"]).unwrap();
    assert_eq!(other.translate(&t, a), Some(PiiKind::Custom(CustomKindId::new(1))));
    assert_eq!(t.translate(&other, other.find("other").unwrap()), None);
}

#[test]
fn layers_with_different_custom_kinds_share_no_state() {
    let first = CustomKinds::from_names(["ticket"]).unwrap();
    let second = CustomKinds::from_names(["badge", "ticket"]).unwrap();
    let mut a = Pii::with_custom_kinds(ViewId::SOURCE, first);
    let mut b = Pii::with_custom_kinds(ViewId::SOURCE, second);
    let id0 = PiiKind::Custom(CustomKindId::new(0));
    a.push(m(0, 3, id0, "x"));
    b.push(m(0, 3, id0, "x"));
    assert_eq!(a.kind_name(id0), Some("ticket"));
    assert_eq!(b.kind_name(id0), Some("badge"), "the same id names another kind in another layer");
    assert_eq!(a.check(), Ok(()));
    assert_eq!(b.check(), Ok(()));
    let mut c = Pii::new(ViewId::SOURCE);
    c.push(m(0, 3, id0, "x"));
    let err = c.check().unwrap_err();
    assert!(err.contains("not in the kind table"), "{err}");
    assert!(!err.contains("\"x\""), "no value in the error");
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
}

#[test]
fn check_wants_priority_order_not_id_order() {
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Email, "a"));
    p.push(m(0, 9, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(m(0, 5, PiiKind::Phone, "a"));
    p.push(m(0, 5, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    // A custom kind sorts after every built-in kind on one span.
    let t = CustomKinds::from_names(["ticket"]).unwrap();
    let ticket = t.find("ticket").unwrap();
    let mut p = Pii::with_custom_kinds(ViewId::SOURCE, t.clone());
    p.push(m(0, 5, PiiKind::Phone, "a"));
    p.push(m(0, 5, ticket, "a"));
    assert_eq!(p.check(), Ok(()));
    let mut p = Pii::with_custom_kinds(ViewId::SOURCE, t);
    p.push(m(0, 5, ticket, "a"));
    p.push(m(0, 5, PiiKind::Phone, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    assert_eq!(
        mention_order(0, 5, PiiKind::Email),
        (0, std::cmp::Reverse(5), (PiiKind::Email.priority(), PiiKind::Email.code()))
    );
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

#[test]
fn debug_of_a_mention_and_of_the_value_column_withholds_the_value() {
    let planted = "planted-4111111111111111";
    let mention = m(0, 5, PiiKind::Email, planted);
    let shown = format!("{mention:?} {mention:#?}");
    assert!(!shown.contains(planted), "{shown}");
    assert!(shown.contains(crate::WITHHELD), "{shown}");
    let mut p = Pii::new(ViewId::SOURCE);
    p.push(mention);
    let column = format!("{:?}", p.normalized);
    assert!(!column.contains(planted), "{column}");
    assert_eq!(p.get(Ix::new(0)).normalized, planted);
}
