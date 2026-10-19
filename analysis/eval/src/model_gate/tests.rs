use super::*;

pub(super) fn f(tp: u64, selected: u64, target: u64) -> Counts {
    Counts { true_positives: tp, selected, target }
}

/// `n` sentences of 10 tokens; the reference gets 9 right on each, the
/// candidate `right(i)` on sentence i.
pub(super) fn tagging(n: u64, right: impl Fn(u64) -> u64) -> Vec<PairedUnit> {
    (0..n)
        .map(|i| PairedUnit { candidate: Counts::accuracy(right(i), 10), reference: Counts::accuracy(9, 10) })
        .collect()
}

/// A held-out set scored `scoring`, with nothing skipped.
pub(super) fn set(name: &str, scoring: Scoring) -> HeldOutSet {
    HeldOutSet {
        name: name.into(),
        corpus_sha256: "a".repeat(64),
        samples_sha256: "b".repeat(64),
        skipped_sentences: 0,
        scoring,
    }
}

/// The tag scoring of `units`, `left_out` tokens left out.
pub(super) fn tags_of(units: &[PairedUnit], left_out: u64) -> Scoring {
    let scored: u64 = units.iter().map(|u| u.candidate.target).sum();
    let mapping = TagMapping { held_out: "(".into(), reference: "-LRB-".into() };
    Scoring::Tags(TagScoring { tokens: scored + left_out, left_out, reference_mappings: vec![mapping] })
}

pub(super) fn record_text(units: Vec<PairedUnit>) -> String {
    let set = set("s", tags_of(&units, 0));
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    ModelGateRecord::new(Task::PartOfSpeechTagger, set, model("t"), model("r"), units).unwrap().to_string()
}

fn typed(types: &[&str], renamed: &[(&str, &str)], inputs: &[(&str, &str)], spans: u64, left_out: u64) -> Scoring {
    Scoring::TypedSpans(TypedSpanScoring {
        types: types.iter().map(|t| t.to_string()).collect(),
        renamed: renamed
            .iter()
            .map(|&(held_out, scored)| TypeMapping { held_out: held_out.into(), scored: scored.into() })
            .collect(),
        reference_inputs: inputs
            .iter()
            .map(|&(held_out, reference)| TagMapping { held_out: held_out.into(), reference: reference.into() })
            .collect(),
        spans,
        left_out,
    })
}

pub(super) fn record() -> ModelGateRecord {
    let set = set("corpus-1.0 held out", Scoring::Spans);
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    ModelGateRecord::new(Task::Tokenizer, set, model("trained"), model("published"), tagging(40, |i| 8 + i % 3))
        .unwrap()
}

/// `n` held-out texts, the reference right on all but every tenth and
/// the candidate on all but every `miss`th.
pub(super) fn labels(n: usize, miss: usize) -> Vec<PairedUnit> {
    (0..n)
        .map(|i| PairedUnit {
            candidate: Counts::accuracy(u64::from(i % miss != 0), 1),
            reference: Counts::accuracy(u64::from(i % 10 != 0), 1),
        })
        .collect()
}

pub(super) fn language_record(units: Vec<PairedUnit>, scoring: Scoring) -> ModelGateRecord {
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    ModelGateRecord::new(Task::LanguageDetector, set("s", scoring), model("t"), model("r"), units).unwrap()
}

pub(super) fn labels_of(units: &[PairedUnit]) -> Scoring {
    let mapping = LabelMapping { held_out: "est".into(), reference: "ekk".into() };
    Scoring::Labels(LabelScoring { samples: units.len() as u64, reference_mappings: vec![mapping] })
}

#[test]
fn a_typed_span_record_scores_its_types_and_counts_the_spans_scored() {
    assert_eq!(Task::NameFinder.measure(), Measure::FMeasure);
    assert_eq!(Task::Chunker.measure(), Measure::FMeasure);
    assert_eq!(Task::NameFinder.scoring(), ScoringKind::TypedSpans);
    assert_eq!(Task::Chunker.scoring(), ScoringKind::TypedSpans);
    // 40 sentences of two names each, of which the candidate finds one.
    let units: Vec<PairedUnit> = (0..40).map(|_| PairedUnit { candidate: f(1, 2, 2), reference: f(2, 2, 2) }).collect();
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    let names =
        |task, scoring| ModelGateRecord::new(task, set("s", scoring), model("t"), model("r"), units.clone()).unwrap();
    let r = names(Task::NameFinder, typed(&["organization"], &[("org", "organization")], &[], 80, 7));
    r.check().unwrap();
    assert_eq!(r.measure, Measure::FMeasure);
    assert!(!r.result.pass);
    let json = r.to_json();
    assert!(json.contains("\"task\": \"name_finder\""), "{json}");
    let scoring = r#""typed_spans": {
        "types": [
          "organization"
        ],
        "renamed": [
          {
            "held_out": "org",
            "scored": "organization"
          }
        ],
        "reference_inputs": [],
        "spans": 80,
        "left_out": 7
      }"#;
    assert!(json.contains(scoring), "{json}");
    let back: ModelGateRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
    // A chunker's reference given `:` for two held-out tags.
    let chunks = names(Task::Chunker, typed(&["NP", "VP"], &[], &[("-", ":"), ("...", ":")], 80, 0));
    chunks.check().unwrap();
    let inputs = r#""reference_inputs": [
          {
            "held_out": "-",
            "reference": ":"
          },
          {
            "held_out": "...",
            "reference": ":"
          }
        ],"#;
    assert!(chunks.to_json().contains(inputs), "{}", chunks.to_json());
    assert_eq!(serde_json::from_str::<ModelGateRecord>(&chunks.to_json()).unwrap(), chunks);

    // The spans scored must be the references of each model's units.
    let mut more = r.clone();
    more.set.scoring = typed(&["organization"], &[], &[], 81, 0);
    assert_eq!(more.check(), Err(ModelGateError::ScoredSpans { spans: 81, scored: 80 }));
    let refused = |scoring: Scoring| {
        let mut bad = r.clone();
        bad.set.scoring = scoring;
        bad.check().unwrap_err()
    };
    assert_eq!(refused(typed(&[], &[], &[], 80, 0)), ModelGateError::NoSpanTypes);
    assert_eq!(
        refused(typed(&["person", ""], &[], &[], 80, 0)),
        ModelGateError::Field { field: "set.scoring.typed_spans.types", value: String::new() }
    );
    assert_eq!(
        refused(typed(&["organization"], &[("", "organization")], &[], 80, 0)),
        ModelGateError::Field { field: "set.scoring.typed_spans.renamed", value: String::new() }
    );
    assert_eq!(
        refused(typed(&["NP"], &[], &[("", ":")], 80, 0)),
        ModelGateError::Field { field: "set.scoring.typed_spans.reference_inputs", value: String::new() }
    );
    let e = refused(typed(&["person", "organization", "person"], &[], &[], 80, 0));
    assert_eq!(e, ModelGateError::DuplicateSpanType { span_type: "person".into() });
    assert_eq!(e.to_string(), "the span type \"person\" is scored twice");
    let twice = [("org", "organization"), ("pers", "person"), ("org", "person")];
    let e = refused(typed(&["organization", "person"], &twice, &[], 80, 0));
    assert_eq!(e, ModelGateError::DuplicateRenaming { held_out: "org".into() });
    assert_eq!(e.to_string(), "the held-out span type \"org\" is renamed twice");
    let e = refused(typed(&["organization"], &[("org", "company")], &[], 80, 0));
    assert_eq!(e, ModelGateError::RenamedToUnscored { held_out: "org".into(), scored: "company".into() });
    assert_eq!(e.to_string(), "the held-out span type \"org\" is renamed \"company\", a type not scored");
    let e = refused(typed(&["organization", "person"], &[("person", "organization")], &[], 80, 0));
    assert_eq!(e, ModelGateError::RenamedScoredType { held_out: "person".into() });
    // A held-out tag the reference is given another tag for, twice
    // (even the same tag twice), or for itself.
    for inputs in [[("-", ":"), ("-", ",")], [("-", ":"), ("-", ":")]] {
        let e = refused(typed(&["NP"], &[], &inputs, 80, 0));
        assert_eq!(e, ModelGateError::DuplicateReferenceInput { held_out: "-".into() });
        assert_eq!(e.to_string(), "the reference is given another tag for the held-out tag \"-\" twice");
    }
    let e = refused(typed(&["NP"], &[], &[("-", ":"), (";", ";")], 80, 0));
    assert_eq!(e, ModelGateError::UnchangedReferenceInput { tag: ";".into() });
    // Typed spans are the scoring of names and chunks only, and names
    // are scored as typed spans.
    let mut untyped = r.clone();
    untyped.set.scoring = Scoring::Spans;
    assert_eq!(
        untyped.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::NameFinder, scoring: ScoringKind::Spans })
    );
    let mut tokens = record();
    tokens.set.scoring = typed(&["x"], &[], &[], 0, 0);
    assert_eq!(
        tokens.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::Tokenizer, scoring: ScoringKind::TypedSpans })
    );
}

#[test]
fn a_record_s_scoring_fits_its_measure_and_counts_the_scored_tokens() {
    let units = tagging(40, |i| 8 + i % 3);
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    let tagger = |scoring| {
        ModelGateRecord::new(Task::PartOfSpeechTagger, set("s", scoring), model("t"), model("r"), units.clone())
            .unwrap()
    };
    let r = tagger(tags_of(&units, 3));
    r.check().unwrap();
    assert_eq!(r.set.skipped_sentences, 0);
    let json = r.to_json();
    assert!(
        json.contains("\"scoring\": {\n      \"tags\": {\n        \"tokens\": 403,\n        \"left_out\": 3,"),
        "{json}"
    );
    assert!(json.contains("\"held_out\": \"(\",\n            \"reference\": \"-LRB-\""), "{json}");
    let back: ModelGateRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);

    // The tokens less those left out must be what each model's units
    // count.
    let mut more = r.clone();
    let Scoring::Tags(t) = &mut more.set.scoring else { unreachable!() };
    t.tokens += 1;
    assert_eq!(more.check(), Err(ModelGateError::ScoredTokens { tokens: 404, left_out: 3, scored: 400 }));
    let mut over = r.clone();
    let Scoring::Tags(t) = &mut over.set.scoring else { unreachable!() };
    t.left_out = t.tokens + 1;
    assert!(matches!(over.check(), Err(ModelGateError::ScoredTokens { .. })));
    // Tag mappings must be non-empty, distinct, and not map a tag to itself.
    for (held_out, reference) in [("", "-LRB-"), ("(", ""), ("NN", "NN")] {
        let mut bad_map = r.clone();
        let Scoring::Tags(t) = &mut bad_map.set.scoring else { unreachable!() };
        t.reference_mappings = vec![TagMapping { held_out: held_out.into(), reference: reference.into() }];
        assert_eq!(
            bad_map.check(),
            Err(ModelGateError::TagMapping { held_out: held_out.into(), reference: reference.into() })
        );
    }
    let mut dup_map = r.clone();
    let Scoring::Tags(t) = &mut dup_map.set.scoring else { unreachable!() };
    t.reference_mappings = vec![
        TagMapping { held_out: "(".into(), reference: "-LRB-".into() },
        TagMapping { held_out: "(".into(), reference: "SYM".into() },
    ];
    assert_eq!(dup_map.check(), Err(ModelGateError::TagMapping { held_out: "(".into(), reference: "SYM".into() }));

    // Spans are an F-measure's scoring, tags an accuracy's.
    assert_eq!(
        tagger(Scoring::Spans).check(),
        Err(ModelGateError::ScoringMismatch { task: Task::PartOfSpeechTagger, scoring: ScoringKind::Spans })
    );
    let mut spans = record();
    spans.set.scoring = tags_of(&spans.units, 0);
    assert_eq!(
        spans.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::Tokenizer, scoring: ScoringKind::Tags })
    );
    assert_eq!(serde_json::to_string(&Scoring::Spans).unwrap(), "\"spans\"");
    // A lemmatizer is scored as a tagger, tokens left out and counted
    // the same way, with no mapping.
    let lemmas = TagScoring {
        tokens: units.iter().map(|u| u.candidate.target).sum::<u64>() + 3,
        left_out: 3,
        reference_mappings: vec![],
    };
    let lemmatizer = |scoring| {
        ModelGateRecord::new(Task::Lemmatizer, set("s", scoring), model("t"), model("r"), units.clone()).unwrap()
    };
    assert_eq!(Task::Lemmatizer.measure(), Measure::Accuracy);
    lemmatizer(Scoring::Tags(lemmas.clone())).check().unwrap();
    assert_eq!(
        lemmatizer(Scoring::Spans).check(),
        Err(ModelGateError::ScoringMismatch { task: Task::Lemmatizer, scoring: ScoringKind::Spans })
    );
    let miscounted = TagScoring { left_out: 2, ..lemmas.clone() };
    assert!(matches!(lemmatizer(Scoring::Tags(miscounted)).check(), Err(ModelGateError::ScoredTokens { .. })));
    // The tagger's record with its mapping passes; as a lemmatizer's it
    // is refused.
    assert_eq!(lemmatizer(tags_of(&units, 3)).check(), Err(ModelGateError::LemmatizerMappings { mappings: 1 }));
    // Unit sums exceeding u64::MAX cannot falsely match tokens = u64::MAX.
    let max_unit =
        PairedUnit { candidate: Counts::accuracy(u64::MAX, u64::MAX), reference: Counts::accuracy(u64::MAX, u64::MAX) };
    let max_units = vec![max_unit; MIN_UNITS];
    let max_tags = Scoring::Tags(TagScoring { tokens: u64::MAX, left_out: 0, reference_mappings: vec![] });
    let max_rec =
        ModelGateRecord::new(Task::Lemmatizer, set("s", max_tags), model("t"), model("r"), max_units).unwrap();
    assert_eq!(max_rec.check(), Err(ModelGateError::ScoredTokens { tokens: u64::MAX, left_out: 0, scored: u64::MAX }));
}

#[test]
fn a_language_detector_is_scored_one_unit_per_text() {
    assert_eq!(Task::LanguageDetector.measure(), Measure::Accuracy);
    // 2,000 texts: the reference wrong on one in ten, the candidate on
    // one in twenty, so the candidate is 5 points ahead.
    let units = labels(2000, 20);
    let r = language_record(units.clone(), labels_of(&units));
    r.check().unwrap();
    assert!((r.result.candidate.score - 95.0).abs() < 1e-9, "{:?}", r.result);
    assert!((r.result.reference.score - 90.0).abs() < 1e-9, "{:?}", r.result);
    assert!(r.result.pass);
    let json = r.to_json();
    assert!(json.contains("\"task\": \"language_detector\""), "{json}");
    assert!(json.contains("\"labels\": {\n        \"samples\": 2000,"), "{json}");
    let back: ModelGateRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);

    // One unit per sample, each of one sample.
    let mut more = r.clone();
    let Scoring::Labels(l) = &mut more.set.scoring else { unreachable!() };
    l.samples += 1;
    assert_eq!(more.check(), Err(ModelGateError::ScoredSamples { samples: 2001, units: 2000, unit: None }));
    let mut two = units.clone();
    two[7] = PairedUnit { candidate: Counts::accuracy(2, 2), reference: Counts::accuracy(1, 2) };
    let two = language_record(two.clone(), labels_of(&two));
    assert_eq!(two.check(), Err(ModelGateError::ScoredSamples { samples: 2000, units: 2000, unit: Some(7) }));
    // Labels are a language detector's scoring, and tags a tagger's.
    let tags = language_record(units.clone(), tags_of(&units, 0));
    assert_eq!(
        tags.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::LanguageDetector, scoring: ScoringKind::Tags })
    );
    let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
    let tagger =
        ModelGateRecord::new(Task::PartOfSpeechTagger, set("s", labels_of(&units)), model("t"), model("r"), units)
            .unwrap();
    assert_eq!(
        tagger.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::PartOfSpeechTagger, scoring: ScoringKind::Labels })
    );
    let mut spans = record();
    spans.set.scoring = labels_of(&spans.units);
    assert_eq!(
        spans.check(),
        Err(ModelGateError::ScoringMismatch { task: Task::Tokenizer, scoring: ScoringKind::Labels })
    );
}

#[test]
fn a_label_scoring_skips_nothing_and_maps_each_label_once_to_another() {
    let units = labels(40, 20);
    let r = language_record(units.clone(), labels_of(&units));
    r.check().unwrap();
    let mut skipped = r.clone();
    skipped.set.skipped_sentences = 2;
    assert_eq!(skipped.check(), Err(ModelGateError::SkippedSamples { skipped: 2 }));
    let mapping =
        |held_out: &str, reference: &str| LabelMapping { held_out: held_out.into(), reference: reference.into() };
    let with = |mappings: Vec<LabelMapping>| {
        let mut m = r.clone();
        let Scoring::Labels(l) = &mut m.set.scoring else { unreachable!() };
        l.reference_mappings = mappings;
        m.check()
    };
    assert_eq!(with(vec![]), Ok(()));
    assert_eq!(with(vec![mapping("est", "ekk"), mapping("est", "et"), mapping("lvs", "lav")]), Ok(()));
    let refused = |held_out: &str, reference: &str| {
        Err(ModelGateError::LabelMapping { held_out: held_out.into(), reference: reference.into() })
    };
    assert_eq!(with(vec![mapping("", "ekk")]), refused("", "ekk"));
    assert_eq!(with(vec![mapping("est", "")]), refused("est", ""));
    assert_eq!(with(vec![mapping("lvs", "lav"), mapping("est", "est")]), refused("est", "est"));
    assert_eq!(with(vec![mapping("est", "ekk"), mapping("lvs", "lav"), mapping("est", "ekk")]), refused("est", "ekk"));
}
