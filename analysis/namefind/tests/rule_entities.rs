//! Tests for deterministic rule-to-`Entities` projection (`Temporals`,
//! `Times`, `Money`, and `%`-`Quantities`) and consensus merging with
//! learned NER heads (including MASC `date`).

use turbo_analysis::annotate::Bound;
use turbo_analysis::{
    Analyzer, Document, Evidence, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{
    Currency, Date, Entities, Modifier, Money, MoneyMention, Origin, PERCENT, Quantities, QuantityMention,
    TemporalMention, TemporalValue, Temporals, TimeGrain, TimeKind, TimeMention, Times, Tokens,
};
use turbo_namefind::{
    ConsensusEntityMerger, DATE_ENTITY, DateTimeEntityKind, EntityMergePolicy, MONEY_ENTITY, PERCENT_ENTITY,
    RULE_ENTITY_KINDS, RuleEntitiesAnnotator, RuleEntityProjector, TIME_ENTITY, TimeEntityFilter, Vote, consensus,
    money_to_entities, quantities_to_entities, rule_entities, span_token_range, temporals_to_entities,
    times_to_entities, times_to_entities_with,
};

fn tokens_from(text: &str, parts: &[&str]) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut pos = 0usize;
    for &part in parts {
        let rel = text[pos..].find(part).unwrap_or_else(|| panic!("missing token {part:?} after byte {pos}"));
        let start = pos + rel;
        let end = start + part.len();
        spans.push(Span::new(start as u32, end as u32));
        pos = end;
    }
    Tokens { spans, sentence: None }
}

fn rows<'a>(text: &'a str, e: &'a Entities) -> Vec<(&'a str, usize, usize, &'a str, f64)> {
    (0..e.len())
        .map(|i| {
            let s = e.spans.get(turbo_analysis::Ix::new(i as u32));
            let r = e.tokens[i];
            (&text[s.start as usize..s.end as usize], r.start.get(), r.end.get(), e.kinds.label(e.kind[i]), e.score[i])
        })
        .collect()
}

fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "r1".into(), sha256: "a".repeat(64) }
}

fn usd() -> Currency {
    Currency::new(*b"USD").unwrap()
}

#[test]
fn projects_temporals_times_money_and_percent_quantities_to_entities() {
    // Tokens:
    // 0:On 1:July 2:14 3:, 4:2026 5:at 6:3:30pm 7:for 8:2 9:hours 10:,
    // 11:Acme 12:paid 13:$1.2M 14:( 15:up 16:15% 17:on 18:80 19:kg 20:) 21:.
    let text = "On July 14, 2026 at 3:30pm for 2 hours, Acme paid $1.2M (up 15% on 80 kg).";
    let toks = tokens_from(
        text,
        &[
            "On", "July", "14", ",", "2026", "at", "3:30pm", "for", "2", "hours", ",", "Acme", "paid", "$1.2M", "(",
            "up", "15%", "on", "80", "kg", ")", ".",
        ],
    );

    let mut temporals = Temporals::new(ViewId::SOURCE);
    temporals.push(TemporalMention {
        span: Span::new(3, 16),
        value: TemporalValue::Day(Date::new(2026, 7, 14).unwrap()),
        origin: Origin::Absolute,
    });
    let date_ents = temporals_to_entities(&temporals, &toks).unwrap();
    date_ents.check().unwrap();
    assert_eq!(rows(text, &date_ents), [("July 14, 2026", 1, 5, DATE_ENTITY, 0.0)]);

    let mut times = Times::new(ViewId::SOURCE);
    times.push(TimeMention {
        span: Span::new(3, 16),
        kind: TimeKind::Date,
        grain: TimeGrain::Day,
        modifier: Modifier::NONE,
        timex: "2026-07-14",
        start: Some(1_784_012_400),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    times.push(TimeMention {
        span: Span::new(20, 26),
        kind: TimeKind::Time,
        grain: TimeGrain::Minute,
        modifier: Modifier::NONE,
        timex: "T15:30",
        start: Some(55_800),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    times.push(TimeMention {
        span: Span::new(31, 38),
        kind: TimeKind::Duration,
        grain: TimeGrain::Hour,
        modifier: Modifier::NONE,
        timex: "PT2H",
        start: None,
        end: None,
        duration_ms: Some(7_200_000),
        ambiguous: false,
    });
    let time_ents = times_to_entities(&times, &toks).unwrap();
    time_ents.check().unwrap();
    assert_eq!(rows(text, &time_ents), [("July 14, 2026", 1, 5, DATE_ENTITY, 0.0), ("3:30pm", 6, 7, TIME_ENTITY, 0.0)]);

    let mut money = Money::new(ViewId::SOURCE);
    money.push(MoneyMention {
        span: Span::new(50, 55),
        negative: false,
        coefficient: "1200000",
        scale: 0,
        currency: usd(),
    });
    let money_ents = money_to_entities(&money, &toks).unwrap();
    money_ents.check().unwrap();
    assert_eq!(rows(text, &money_ents), [("$1.2M", 13, 14, MONEY_ENTITY, 0.0)]);

    let mut quantities = Quantities::new(ViewId::SOURCE);
    quantities.push(QuantityMention {
        span: Span::new(60, 63),
        negative: false,
        coefficient: "15",
        scale: 0,
        unit: PERCENT,
    });
    quantities.push(QuantityMention {
        span: Span::new(67, 72),
        negative: false,
        coefficient: "80",
        scale: 0,
        unit: "kg",
    });
    let pct_ents = quantities_to_entities(&quantities, &toks).unwrap();
    pct_ents.check().unwrap();
    assert_eq!(rows(text, &pct_ents), [("15%", 16, 17, PERCENT_ENTITY, 0.0)]);

    let combined = rule_entities(
        &toks,
        Some(&temporals),
        Some(&times),
        Some(&money),
        Some(&quantities),
        TimeEntityFilter::DatesAndTimes,
    )
    .unwrap();
    combined.check().unwrap();
    assert_eq!(
        rows(text, &combined),
        [
            ("July 14, 2026", 1, 5, DATE_ENTITY, 0.0),
            ("3:30pm", 6, 7, TIME_ENTITY, 0.0),
            ("$1.2M", 13, 14, MONEY_ENTITY, 0.0),
            ("15%", 16, 17, PERCENT_ENTITY, 0.0),
        ]
    );
}

#[test]
fn time_entity_filters_and_view_checks_behave_deterministically() {
    let text = "2026-07-14 09:30 tomorrow at 8am every Monday";
    let toks = tokens_from(text, &["2026-07-14", "09:30", "tomorrow", "at", "8am", "every", "Monday"]);
    let mut times = Times::new(ViewId::SOURCE);
    times.push(TimeMention {
        span: Span::new(0, 10),
        kind: TimeKind::Date,
        grain: TimeGrain::Day,
        modifier: Modifier::NONE,
        timex: "2026-07-14",
        start: Some(0),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    times.push(TimeMention {
        span: Span::new(11, 16),
        kind: TimeKind::Time,
        grain: TimeGrain::Minute,
        modifier: Modifier::NONE,
        timex: "T09:30",
        start: Some(34_200),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    times.push(TimeMention {
        span: Span::new(17, 32),
        kind: TimeKind::DateTime,
        grain: TimeGrain::Hour,
        modifier: Modifier::NONE,
        timex: "2026-07-15T08",
        start: Some(86_400),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    times.push(TimeMention {
        span: Span::new(33, 45),
        kind: TimeKind::Set,
        grain: TimeGrain::Week,
        modifier: Modifier::NONE,
        timex: "XXXX-WXX-1",
        start: None,
        end: None,
        duration_ms: None,
        ambiguous: false,
    });

    let as_time = times_to_entities_with(&times, &toks, TimeEntityFilter::DatesAndTimes).unwrap();
    assert_eq!(
        rows(text, &as_time),
        [
            ("2026-07-14", 0, 1, DATE_ENTITY, 0.0),
            ("09:30", 1, 2, TIME_ENTITY, 0.0),
            ("tomorrow at 8am", 2, 5, TIME_ENTITY, 0.0),
        ]
    );

    let as_date =
        times_to_entities_with(&times, &toks, TimeEntityFilter::DatesAndTimesWith(DateTimeEntityKind::Date)).unwrap();
    assert_eq!(
        rows(text, &as_date),
        [
            ("2026-07-14", 0, 1, DATE_ENTITY, 0.0),
            ("09:30", 1, 2, TIME_ENTITY, 0.0),
            ("tomorrow at 8am", 2, 5, DATE_ENTITY, 0.0),
        ]
    );

    let skip_dt =
        times_to_entities_with(&times, &toks, TimeEntityFilter::DatesAndTimesWith(DateTimeEntityKind::Skip)).unwrap();
    assert_eq!(rows(text, &skip_dt), [("2026-07-14", 0, 1, DATE_ENTITY, 0.0), ("09:30", 1, 2, TIME_ENTITY, 0.0)]);

    let times_only = times_to_entities_with(&times, &toks, TimeEntityFilter::TimesOnly).unwrap();
    assert_eq!(
        rows(text, &times_only),
        [("09:30", 1, 2, TIME_ENTITY, 0.0), ("tomorrow at 8am", 2, 5, TIME_ENTITY, 0.0)]
    );

    let clock_only = times_to_entities_with(&times, &toks, TimeEntityFilter::ClockTimesOnly).unwrap();
    assert_eq!(rows(text, &clock_only), [("09:30", 1, 2, TIME_ENTITY, 0.0)]);

    let dates_only = times_to_entities_with(&times, &toks, TimeEntityFilter::DatesOnly).unwrap();
    assert_eq!(rows(text, &dates_only), [("2026-07-14", 0, 1, DATE_ENTITY, 0.0)]);

    assert_eq!(span_token_range(&toks, Span::new(10, 10)), None);
    assert_eq!(span_token_range(&toks, Span::new(10, 11)), None);

    let wrong_view = Tokens { spans: Spans::new(ViewId(7)), sentence: None };
    assert!(times_to_entities(&times, &wrong_view).is_err());
    assert!(temporals_to_entities(&Temporals::new(ViewId::SOURCE), &wrong_view).is_err());
    assert!(money_to_entities(&Money::new(ViewId::SOURCE), &wrong_view).is_err());
    assert!(quantities_to_entities(&Quantities::new(ViewId::SOURCE), &wrong_view).is_err());
}

#[test]
fn projector_preserves_source_provenance_and_merges_with_masc_date_head() {
    // Tokens:
    // 0:Ada 1:Lovelace 2:visited 3:London 4:on 5:July 6:14 7:, 8:2026
    // 9:at 10:3:30pm 11:and 12:Next 13:Spring 14:paid 15:$250 16:( 17:12.5% 18:) 19:.
    let text = "Ada Lovelace visited London on July 14, 2026 at 3:30pm and Next Spring paid $250 (12.5%).";
    let toks = tokens_from(
        text,
        &[
            "Ada", "Lovelace", "visited", "London", "on", "July", "14", ",", "2026", "at", "3:30pm", "and", "Next",
            "Spring", "paid", "$250", "(", "12.5%", ")", ".",
        ],
    );

    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, toks, Provenance::by("turbo-tokenize:simple")).unwrap();

    let mut temporals = Temporals::new(ViewId::SOURCE);
    temporals.push(TemporalMention {
        span: Span::new(31, 44),
        value: TemporalValue::Day(Date::new(2026, 7, 14).unwrap()),
        origin: Origin::Absolute,
    });
    doc.add(
        &Key::DEFAULT,
        temporals,
        Provenance { version: "1", config: 42, ..Provenance::by("turbo-numeric:temporal") },
    )
    .unwrap();

    let mut times = Times::new(ViewId::SOURCE);
    times.push(TimeMention {
        span: Span::new(48, 54),
        kind: TimeKind::Time,
        grain: TimeGrain::Minute,
        modifier: Modifier::NONE,
        timex: "T15:30",
        start: Some(55_800),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    doc.add(&Key::DEFAULT, times, Provenance { version: "1", config: 7, ..Provenance::by("turbo-grammar:times") })
        .unwrap();

    let mut money = Money::new(ViewId::SOURCE);
    money.push(MoneyMention {
        span: Span::new(76, 80),
        negative: false,
        coefficient: "250",
        scale: 0,
        currency: usd(),
    });
    doc.add(&Key::DEFAULT, money, Provenance { version: "1", config: 9, ..Provenance::by("turbo-numeric:money") })
        .unwrap();

    let mut quantities = Quantities::new(ViewId::SOURCE);
    quantities.push(QuantityMention {
        span: Span::new(82, 87),
        negative: false,
        coefficient: "125",
        scale: 1,
        unit: PERCENT,
    });
    doc.add(
        &Key::DEFAULT,
        quantities,
        Provenance { version: "1", config: 11, ..Provenance::by("turbo-numeric:quantities") },
    )
    .unwrap();

    // Simulated MASC NER head ("person", "location", "organization", "date"):
    // - "Ada Lovelace" (0..2, person, 0.96)
    // - "London" (3..4, location, 0.94)
    // - Partial overlapping date "July 14" (5..7, date, 0.88): overlaps the
    //   deterministic rule's full "July 14, 2026" (5..9), so the rule's full
    //   span wins and absorbs this vote and its 0.88 model confidence.
    // - Disjoint learned date "Next Spring" (12..14, date, 0.79): not matched
    //   by TemporalExtractor, so the MASC head fills the gap.
    // - False-positive "organization" on "$250" (15..16, organization, 0.61):
    //   overlaps the deterministic "money" rule span and is suppressed.
    let masc_kinds = TagSet::new("ner", &["person", "location", "organization", "date"]);
    let mut masc_spans = Spans::new(ViewId::SOURCE);
    masc_spans.push(Span::new(0, 12));
    masc_spans.push(Span::new(21, 27));
    masc_spans.push(Span::new(31, 38));
    masc_spans.push(Span::new(59, 70));
    masc_spans.push(Span::new(76, 80));
    let masc_entities = Entities {
        spans: masc_spans,
        tokens: vec![
            IxRange::new(0, 2),
            IxRange::new(3, 4),
            IxRange::new(5, 7),
            IxRange::new(12, 14),
            IxRange::new(15, 16),
        ],
        kind: vec![Tag(0), Tag(1), Tag(3), Tag(3), Tag(2)],
        kinds: masc_kinds,
        score: vec![0.96, 0.94, 0.88, 0.79, 0.61],
    };
    doc.add(
        &Key::named("masc"),
        masc_entities,
        Provenance {
            reads: vec![Key::<Tokens>::DEFAULT.id()],
            model: Some(model("masc-ne")),
            ..Provenance::by("turbo-tagger:masc-ne")
        },
    )
    .unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Times>::DEFAULT)
        .given(Key::<Money>::DEFAULT)
        .given(Key::<Quantities>::DEFAULT)
        .given(Key::<Entities>::named("masc"))
        .step(RuleEntityProjector::money(Key::DEFAULT, Key::named("money")))
        .step(RuleEntityProjector::quantities(Key::DEFAULT, Key::named("percent")))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(RuleEntityProjector::times_with_filter(Key::DEFAULT, Key::named("times"), TimeEntityFilter::TimesOnly))
        .step(ConsensusEntityMerger::new(
            vec![
                Key::named("money"),
                Key::named("percent"),
                Key::named("temporal"),
                Key::named("times"),
                Key::named("masc"),
            ],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .build()
        .unwrap();

    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let merged = doc.get_key(&Key::<Entities>::DEFAULT).unwrap();
    merged.check().unwrap();
    assert_eq!(
        rows(text, merged),
        [
            ("Ada Lovelace", 0, 2, "person", 0.96),
            ("London", 3, 4, "location", 0.94),
            ("July 14, 2026", 5, 9, DATE_ENTITY, 0.88),
            ("3:30pm", 10, 11, TIME_ENTITY, 0.0),
            ("Next Spring", 12, 14, DATE_ENTITY, 0.79),
            ("$250", 15, 16, MONEY_ENTITY, 0.0),
            ("12.5%", 17, 18, PERCENT_ENTITY, 0.0),
        ]
    );

    let ev = doc.get_key(&Key::<Evidence<Entities>>::DEFAULT).unwrap();
    ev.verify(&doc).unwrap();
    assert_eq!(ev.records(), 7);
    assert_eq!(ev.producer_of(0).annotator, "turbo-tagger:masc-ne");
    assert_eq!(ev.producer_of(1).annotator, "turbo-tagger:masc-ne");
    // "July 14, 2026": rule producer wins, MASC head adds corroborating vote & raw score 0.88.
    assert_eq!(ev.producer_of(2).annotator, "turbo-numeric:temporal");
    assert_eq!(ev.producer_of(2).model, None);
    assert_eq!(ev.support_of(2).len(), 2);
    assert_eq!(ev.spans_of(2), [Span::new(31, 44), Span::new(31, 38)]);
    assert_eq!(ev.raw_of(2), Some(0.88));
    // "3:30pm": rule-only time mention.
    assert_eq!(ev.producer_of(3).annotator, "turbo-grammar:times");
    assert_eq!(ev.raw_of(3), None);
    // "Next Spring": MASC fallback date mention.
    assert_eq!(ev.producer_of(4).annotator, "turbo-tagger:masc-ne");
    assert_eq!(ev.raw_of(4), Some(0.79));
    // "$250" and "12.5%": rule-only money and percent mentions.
    assert_eq!(ev.producer_of(5).annotator, "turbo-numeric:money");
    assert_eq!(ev.raw_of(5), None);
    assert_eq!(ev.producer_of(6).annotator, "turbo-numeric:quantities");
    assert_eq!(ev.raw_of(6), None);
}

#[test]
fn combined_rule_entities_annotator_runs_in_pipeline() {
    let text = "Revenue hit $50 on 2026-07-14 (up 5%).";
    let toks = tokens_from(text, &["Revenue", "hit", "$50", "on", "2026-07-14", "(", "up", "5%", ")", "."]);
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, toks, Provenance::by("")).unwrap();

    let mut temporals = Temporals::new(ViewId::SOURCE);
    temporals.push(TemporalMention {
        span: Span::new(19, 29),
        value: TemporalValue::Day(Date::new(2026, 7, 14).unwrap()),
        origin: Origin::Absolute,
    });
    doc.add(&Key::DEFAULT, temporals, Provenance::by("")).unwrap();

    let mut money = Money::new(ViewId::SOURCE);
    money.push(MoneyMention { span: Span::new(12, 15), negative: false, coefficient: "50", scale: 0, currency: usd() });
    doc.add(&Key::DEFAULT, money, Provenance::by("")).unwrap();

    let mut quantities = Quantities::new(ViewId::SOURCE);
    quantities.push(QuantityMention {
        span: Span::new(34, 36),
        negative: false,
        coefficient: "5",
        scale: 0,
        unit: PERCENT,
    });
    doc.add(&Key::DEFAULT, quantities, Provenance::by("")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Money>::DEFAULT)
        .given(Key::<Quantities>::DEFAULT)
        .step(Bound::new(RuleEntitiesAnnotator::new()))
        .build()
        .unwrap();

    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let ents = docs[0].get::<Entities>().unwrap();
    assert_eq!(
        rows(text, ents),
        [("$50", 2, 3, MONEY_ENTITY, 0.0), ("2026-07-14", 4, 5, DATE_ENTITY, 0.0), ("5%", 7, 8, PERCENT_ENTITY, 0.0)]
    );
    assert_eq!(RULE_ENTITY_KINDS, ["date", "time", "money", "percent"]);
    let c = consensus(&[ents], EntityMergePolicy::PriorityFallback).unwrap();
    assert_eq!(c.winner, [Vote { input: 0, entity: 0 }, Vote { input: 0, entity: 1 }, Vote { input: 0, entity: 2 }]);
}
