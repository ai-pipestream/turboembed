use super::*;

const SHAPE: &[PiiEvidence] = &[PiiEvidence::Shape];
const LUHN: &[PiiEvidence] = &[PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true }];

/// A mention of the test detector (entry 0 of [`layer`]'s table).
fn m(start: u32, end: u32, kind: PiiKind, normalized: &str) -> PiiMention<'_> {
    PiiMention {
        span: Span::new(start, end),
        kind,
        detector: PiiDetectorRef::new(0),
        evidence: SHAPE,
        score: None,
        normalized,
    }
}

/// An empty layer whose detector table holds the test detector.
fn layer(custom: CustomKinds) -> Pii {
    let mut p = Pii::with_custom_kinds(ViewId::SOURCE, custom);
    p.detector("test", "1").unwrap();
    p
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
}

#[test]
fn tier_derives_from_evidence_by_table() {
    use PiiConfidence::*;
    let label = PiiEvidence::Label { span: Span::new(0, 3), phrase: 0 };
    let passed = PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Mod97, passed: true };
    let failed = PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Mod97, passed: false };
    let deny = PiiEvidence::DenyHit { list: 0, term: 2 };
    let rule = PiiEvidence::Rule { rule: 7 };
    let table: [(&[PiiEvidence], PiiConfidence); 10] = [
        (&[PiiEvidence::Shape], Format),
        (&[rule], Format),
        (&[deny], Format),
        (&[failed], Format),
        (&[PiiEvidence::Shape, label], Labelled),
        (&[failed, label], Labelled),
        (&[passed], Checksum),
        (&[passed, PiiEvidence::Shape], Checksum),
        (&[passed, label], LabelledChecksum),
        (&[label, deny, passed, rule], LabelledChecksum),
    ];
    for (evidence, want) in table {
        assert_eq!(PiiConfidence::derive(evidence), want, "{evidence:?}");
    }
    for a in ChecksumAlgorithm::ALL {
        assert_eq!(ChecksumAlgorithm::from_id(u32::from(a.id())), Some(a));
    }
    assert_eq!(ChecksumAlgorithm::from_id(0), None);
    assert_eq!(ChecksumAlgorithm::from_id(13), None);
    for k in PiiScoreKind::ALL {
        assert_eq!(PiiScoreKind::from_id(u32::from(k.id())), Some(k));
    }
    assert_eq!(PiiScoreKind::from_id(4), None);
}

#[test]
fn check_wants_known_detectors_evidence_and_scores() {
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention { detector: PiiDetectorRef::new(1), ..m(0, 3, PiiKind::Email, "a") });
    assert!(p.check().unwrap_err().contains("detector 1 is not in the table"));
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention { evidence: &[], ..m(0, 3, PiiKind::Email, "a") });
    assert!(p.check().unwrap_err().contains("no evidence"));
    let mut p = layer(CustomKinds::new());
    let empty_label = [PiiEvidence::Label { span: Span::new(4, 4), phrase: 0 }];
    p.push(PiiMention { evidence: &empty_label, ..m(0, 3, PiiKind::Email, "a") });
    assert!(p.check().unwrap_err().contains("label evidence span 4..4 is empty"));
    let model = PiiDetectorRef::new(0);
    let good = PiiScore { value: 0.25, kind: PiiScoreKind::Probability, model, calibrated: true };
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention { score: Some(good), ..m(0, 3, PiiKind::Email, "a") });
    p.push(PiiMention {
        score: Some(PiiScore { value: -3.5, kind: PiiScoreKind::Logit, ..good }),
        ..m(4, 7, PiiKind::Email, "b")
    });
    assert_eq!(p.check(), Ok(()));
    assert_eq!(p.get(Ix::new(0)).score, Some(good));
    assert_eq!(p.get(Ix::new(1)).confidence(), PiiConfidence::Format);
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention { score: Some(PiiScore { value: 1.5, ..good }), ..m(0, 3, PiiKind::Email, "a") });
    assert!(p.check().unwrap_err().contains("outside 0..=1"));
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention {
        score: Some(PiiScore { value: f32::NAN, kind: PiiScoreKind::Logit, ..good }),
        ..m(0, 3, PiiKind::Email, "a")
    });
    assert!(p.check().unwrap_err().contains("not finite"));
    let mut p = layer(CustomKinds::new());
    p.push(PiiMention {
        score: Some(PiiScore { model: PiiDetectorRef::new(9), ..good }),
        ..m(0, 3, PiiKind::Email, "a")
    });
    assert!(p.check().unwrap_err().contains("score model"));
    let mut t = PiiDetectors::new();
    assert_eq!(t.add("", "1"), Err(DetectorError::EmptyId));
    assert_eq!(t.add("a", "1"), Ok(PiiDetectorRef::new(0)));
    assert_eq!(t.add("a", "1"), Ok(PiiDetectorRef::new(0)), "the same entry twice is one");
    assert_eq!(t.add("a", "2"), Ok(PiiDetectorRef::new(1)), "another version is another entry");
    assert_eq!(t.add(&"x".repeat(PiiDetectors::MAX_BYTES + 1), ""), Err(DetectorError::TooLong(129)));
    assert_eq!(t.get(PiiDetectorRef::new(1)), Some(("a", "2")));
    assert_eq!(t.check(PiiDetectorRef::new(2)), Err(DetectorError::Unknown(2)));
}

#[test]
fn candidates_record_dispositions_without_values() {
    use crate::pii_candidates::*;
    let mut c = PiiCandidates::new(ViewId::SOURCE);
    let d = c.detectors.add("test", "1").unwrap();
    let kept = PiiDisposition::Kept { mention: Ix::new(0) };
    let cand = |start, end, kind, disposition| PiiCandidate {
        span: Span::new(start, end),
        kind,
        detector: d,
        evidence: LUHN,
        score: None,
        disposition,
    };
    c.push(cand(0, 9, PiiKind::Card, kept));
    c.push(cand(
        0,
        9,
        PiiKind::Phone,
        PiiDisposition::AllowedByRule { scope: AllowScope::Kind, rule: AllowRule::Term(3) },
    ));
    c.push(cand(2, 6, PiiKind::Phone, PiiDisposition::NestedIn { mention: Ix::new(0) }));
    c.push(cand(12, 20, PiiKind::CaSin, PiiDisposition::RejectedByRule { reason: RejectReason::LabelRequired }));
    c.push(cand(
        30,
        40,
        PiiKind::Email,
        PiiDisposition::AllowedByRule { scope: AllowScope::Global, rule: AllowRule::Pattern(0) },
    ));
    assert_eq!(c.check(), Ok(()));
    assert_eq!(PiiCandidates::ID, "turbo:pii-candidates");
    assert_eq!(c.suppressed_by_kind(), [(PiiKind::Email, 1), (PiiKind::Phone, 1)]);
    assert_eq!(c.get(Ix::new(3)).disposition.mention(), None);
    assert_eq!(c.get(Ix::new(2)).disposition.mention(), Some(Ix::new(0)));
    let mut one = layer(CustomKinds::new());
    one.push(m(0, 9, PiiKind::Card, "123456789"));
    assert_eq!(c.check_against(&one), Ok(()));
    assert!(c.check_against(&layer(CustomKinds::new())).unwrap_err().contains("refers to mention 0 of 0"));
    c.push(cand(1, 5, PiiKind::Email, kept));
    assert!(c.check().unwrap_err().contains("out of order"));
    for r in [RejectReason::LabelRequired, RejectReason::Threshold] {
        assert_eq!(RejectReason::from_id(u32::from(r.id())), Some(r));
    }
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
    let mut a = layer(first);
    let mut b = layer(second);
    let id0 = PiiKind::Custom(CustomKindId::new(0));
    a.push(m(0, 3, id0, "x"));
    b.push(m(0, 3, id0, "x"));
    assert_eq!(a.kind_name(id0), Some("ticket"));
    assert_eq!(b.kind_name(id0), Some("badge"), "the same id names another kind in another layer");
    assert_eq!(a.check(), Ok(()));
    assert_eq!(b.check(), Ok(()));
    let mut c = layer(CustomKinds::new());
    c.push(m(0, 3, id0, "x"));
    let err = c.check().unwrap_err();
    assert!(err.contains("not in the kind table"), "{err}");
    assert!(!err.contains("\"x\""), "no value in the error");
}

#[test]
fn mentions_round_trip_and_may_overlap() {
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 15, PiiKind::Imei, "490154203237518"));
    p.push(PiiMention { evidence: LUHN, ..m(0, 15, PiiKind::Card, "490154203237518") });
    p.push(m(5, 20, PiiKind::Phone, "1234567890"));
    assert_eq!(p.check(), Ok(()));
    assert_eq!(p.len(), 3);
    assert_eq!(p.get(Ix::new(1)).confidence(), PiiConfidence::Checksum);
    assert_eq!(p.get(Ix::new(0)).confidence(), PiiConfidence::Format);
    assert_eq!(p.get(Ix::new(2)).normalized, "1234567890");
    assert_eq!(Pii::ID, "turbo:pii");
}

#[test]
fn check_wants_priority_order_not_id_order() {
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 5, PiiKind::Email, "a"));
    p.push(m(0, 9, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 5, PiiKind::Phone, "a"));
    p.push(m(0, 5, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("out of order"));
    // A custom kind sorts after every built-in kind on one span.
    let t = CustomKinds::from_names(["ticket"]).unwrap();
    let ticket = t.find("ticket").unwrap();
    let mut p = layer(t.clone());
    p.push(m(0, 5, PiiKind::Phone, "a"));
    p.push(m(0, 5, ticket, "a"));
    assert_eq!(p.check(), Ok(()));
    let mut p = layer(t);
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
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 5, PiiKind::Email, "a"));
    p.push(m(0, 5, PiiKind::Email, "b"));
    assert_eq!(p.check(), Ok(()));
    p.push(m(0, 5, PiiKind::Email, "a"));
    assert!(p.check().unwrap_err().contains("repeats"));
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 5, PiiKind::Email, ""));
    assert!(p.check().unwrap_err().contains("no normalized"));
    let mut p = layer(CustomKinds::new());
    p.push(m(3, 3, PiiKind::Email, "x"));
    assert!(p.check().unwrap_err().contains("empty"));
    let mut p = layer(CustomKinds::new());
    p.push(m(0, 3, PiiKind::Email, "x"));
    p.kind.pop();
    assert_eq!(p.check().unwrap_err(), "every column as long as the spans");
}
