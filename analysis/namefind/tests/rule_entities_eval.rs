//! Project-authored gold evaluation for deterministic `date`, `time`, `money`,
//! and `percent` entities, MASC held-out `date` consensus with the `#1730`
//! MASC `date` head, and OntoNotes peer comparison (`DATE`, `TIME`, `MONEY`,
//! `PERCENT` from spaCy `en_core_web_trf` and Stanza OntoNotes NER).

use turbo_analysis::{
    Analyzer, Document, Evidence, Ix, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{
    Currency, Date, Entities, Modifier, Money, MoneyMention, Origin, PERCENT, Quantities, QuantityMention,
    TemporalMention, TemporalValue, Temporals, TimeGrain, TimeKind, TimeMention, Times, Tokens,
};
use turbo_namefind::{
    ConsensusEntityMerger, DATE_ENTITY, EntityMergePolicy, MONEY_ENTITY, PERCENT_ENTITY, RULE_ENTITY_KINDS,
    RuleEntityProjector, TIME_ENTITY, TimeEntityFilter,
};

/// Exact-span evaluation counts and F1 metrics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Prf {
    tp: usize,
    fp: usize,
    fn_count: usize,
}

impl Prf {
    fn precision(self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { 1.0 } else { self.tp as f64 / denom as f64 }
    }

    fn recall(self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 { 1.0 } else { self.tp as f64 / denom as f64 }
    }

    fn f1(self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

/// Gold entity span: `(byte_start, byte_end, tok_start, tok_end, kind)`.
type GoldSpan = (u32, u32, usize, usize, &'static str);

fn extract_spans(entities: &Entities) -> Vec<GoldSpan> {
    (0..entities.len())
        .map(|i| {
            let s = entities.spans.get(Ix::new(i as u32));
            let r = entities.tokens[i];
            let kind = match entities.kinds.label(entities.kind[i]) {
                "date" | "DATE" => DATE_ENTITY,
                "time" | "TIME" => TIME_ENTITY,
                "money" | "MONEY" => MONEY_ENTITY,
                "percent" | "PERCENT" => PERCENT_ENTITY,
                other => panic!("unexpected entity kind in evaluation: {other}"),
            };
            (s.start, s.end, r.start.get(), r.end.get(), kind)
        })
        .collect()
}

fn score_spans(gold: &[GoldSpan], pred: &[GoldSpan], kind_filter: Option<&str>) -> Prf {
    let g: Vec<GoldSpan> = gold.iter().copied().filter(|x| kind_filter.is_none_or(|k| x.4 == k)).collect();
    let mut p: Vec<GoldSpan> = pred.iter().copied().filter(|x| kind_filter.is_none_or(|k| x.4 == k)).collect();
    let mut tp = 0usize;
    for item in &g {
        if let Some(pos) = p.iter().position(|x| x == item) {
            tp += 1;
            p.swap_remove(pos);
        }
    }
    Prf { tp, fp: p.len(), fn_count: g.len() - tp }
}

fn tokenize_exact(text: &str, tok_strs: &[&str]) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut pos = 0usize;
    for &t in tok_strs {
        let off = text[pos..].find(t).unwrap_or_else(|| panic!("token {t:?} not found after {pos} in {text:?}"));
        let start = pos + off;
        let end = start + t.len();
        spans.push(Span::new(start as u32, end as u32));
        pos = end;
    }
    Tokens { spans, sentence: None }
}

#[test]
fn gold_evaluation_baseline_for_date_time_money_and_percent() {
    // Reviewed, hand-authored evaluation document containing all four rule
    // entity types plus distractors ("3 pounds of flour", "80 kg", "room 101").
    //
    // Token index map:
    //  0:On  1:2026-07-14  2:,  3:the  4:board  5:met  6:at  7:9:30am
    //  8:and  9: adjourned  10:at  11:4:15pm  12:.
    // 13:By  14:November  15:3  16:,  17:2026  18:,  19:revenue  20:reached
    // 21:$4.5M  22:and  23:EUR  24:250  25:,  26:up  27:18.5%  28:while
    // 29:operating  30:costs  31:fell  32:4  33:percent  34:on  35:80  36:kg
    // 37:of  38: freight  39:in  40:Q3  41:2026  42:.
    let text = "On 2026-07-14, the board met at 9:30am and adjourned at 4:15pm. \
                By November 3, 2026, revenue reached $4.5M and EUR 250, up 18.5% \
                while operating costs fell 4 percent on 80 kg of freight in Q3 2026.";
    let toks = tokenize_exact(
        text,
        &[
            "On",
            "2026-07-14",
            ",",
            "the",
            "board",
            "met",
            "at",
            "9:30am",
            "and",
            "adjourned",
            "at",
            "4:15pm",
            ".",
            "By",
            "November",
            "3",
            ",",
            "2026",
            ",",
            "revenue",
            "reached",
            "$4.5M",
            "and",
            "EUR",
            "250",
            ",",
            "up",
            "18.5%",
            "while",
            "operating",
            "costs",
            "fell",
            "4",
            "percent",
            "on",
            "80",
            "kg",
            "of",
            "freight",
            "in",
            "Q3",
            "2026",
            ".",
        ],
    );

    let gold: Vec<GoldSpan> = vec![
        (3, 13, 1, 2, DATE_ENTITY),         // "2026-07-14"
        (32, 38, 7, 8, TIME_ENTITY),        // "9:30am"
        (56, 62, 11, 12, TIME_ENTITY),      // "4:15pm"
        (67, 83, 14, 18, DATE_ENTITY),      // "November 3, 2026"
        (101, 106, 21, 22, MONEY_ENTITY),   // "$4.5M"
        (111, 118, 23, 25, MONEY_ENTITY),   // "EUR 250"
        (123, 128, 27, 28, PERCENT_ENTITY), // "18.5%"
        (156, 165, 32, 34, PERCENT_ENTITY), // "4 percent"
        (189, 196, 40, 42, DATE_ENTITY),    // "Q3 2026"
    ];

    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, toks, Provenance::by("turbo-tokenize:simple")).unwrap();

    let mut temporals = Temporals::new(ViewId::SOURCE);
    temporals.push(TemporalMention {
        span: Span::new(3, 13),
        value: TemporalValue::Day(Date::new(2026, 7, 14).unwrap()),
        origin: Origin::Absolute,
    });
    temporals.push(TemporalMention {
        span: Span::new(67, 83),
        value: TemporalValue::Day(Date::new(2026, 11, 3).unwrap()),
        origin: Origin::Absolute,
    });
    temporals.push(TemporalMention {
        span: Span::new(189, 196),
        value: TemporalValue::Quarter { year: 2026, quarter: 3 },
        origin: Origin::Absolute,
    });
    doc.add(&Key::DEFAULT, temporals, Provenance::by("turbo-numeric:temporal")).unwrap();

    let mut times = Times::new(ViewId::SOURCE);
    times.push(TimeMention {
        span: Span::new(32, 38),
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
        span: Span::new(56, 62),
        kind: TimeKind::Time,
        grain: TimeGrain::Minute,
        modifier: Modifier::NONE,
        timex: "T16:15",
        start: Some(58_500),
        end: None,
        duration_ms: None,
        ambiguous: false,
    });
    doc.add(&Key::DEFAULT, times, Provenance::by("turbo-grammar:times")).unwrap();

    let mut money = Money::new(ViewId::SOURCE);
    money.push(MoneyMention {
        span: Span::new(101, 106),
        negative: false,
        coefficient: "4500000",
        scale: 0,
        currency: Currency::new(*b"USD").unwrap(),
    });
    money.push(MoneyMention {
        span: Span::new(111, 118),
        negative: false,
        coefficient: "250",
        scale: 0,
        currency: Currency::new(*b"EUR").unwrap(),
    });
    doc.add(&Key::DEFAULT, money, Provenance::by("turbo-numeric:money")).unwrap();

    let mut quantities = Quantities::new(ViewId::SOURCE);
    quantities.push(QuantityMention {
        span: Span::new(123, 128),
        negative: false,
        coefficient: "185",
        scale: 1,
        unit: PERCENT,
    });
    quantities.push(QuantityMention {
        span: Span::new(156, 165),
        negative: false,
        coefficient: "4",
        scale: 0,
        unit: PERCENT,
    });
    // Distractor quantity ("80 kg") must be ignored by `quantities_to_entities`.
    quantities.push(QuantityMention {
        span: Span::new(169, 174),
        negative: false,
        coefficient: "80",
        scale: 0,
        unit: "kg",
    });
    doc.add(&Key::DEFAULT, quantities, Provenance::by("turbo-numeric:quantities")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Times>::DEFAULT)
        .given(Key::<Money>::DEFAULT)
        .given(Key::<Quantities>::DEFAULT)
        .step(RuleEntityProjector::money(Key::DEFAULT, Key::named("money")))
        .step(RuleEntityProjector::quantities(Key::DEFAULT, Key::named("percent")))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(RuleEntityProjector::times_with_filter(Key::DEFAULT, Key::named("times"), TimeEntityFilter::TimesOnly))
        .step(ConsensusEntityMerger::new(
            vec![Key::named("money"), Key::named("percent"), Key::named("temporal"), Key::named("times")],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .build()
        .unwrap();

    let res = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(res.iter().all(Result::is_ok), "{res:?}");

    let entities = doc.get::<Entities>().unwrap();
    let pred = extract_spans(entities);
    assert_eq!(pred, gold);

    for kind in RULE_ENTITY_KINDS {
        let m = score_spans(&gold, &pred, Some(kind));
        assert_eq!((m.precision(), m.recall(), m.f1()), (1.0, 1.0, 1.0), "failed baseline for {kind}");
    }
    let micro = score_spans(&gold, &pred, None);
    assert_eq!((micro.tp, micro.fp, micro.fn_count, micro.f1()), (9, 0, 0, 1.0));
}

#[test]
fn masc_held_out_date_consensus_and_ontonotes_peer_comparison() {
    // Held-out MASC-style multi-sentence passage with 5 gold `date` spans:
    // 1. "2026-03-15" (ISO dateline, tokens 1..2)
    // 2. "March 18, 2026" (written date, tokens 7..11)
    // 3. "yesterday" (relative to dateline 2026-03-15, tokens 15..16)
    // 4. "Q4 2025" (quarter, tokens 21..23)
    // 5. "Labor Day" (named holiday date in MASC, tokens 27..29)
    let text = "Dateline 2026-03-15: Officials announced on March 18, 2026 that talks \
                began yesterday after the Q4 2025 audit and continue through Labor Day.";
    let toks = tokenize_exact(
        text,
        &[
            "Dateline",
            "2026-03-15",
            ":",
            "Officials",
            "announced",
            "on",
            "March",
            "18",
            ",",
            "2026",
            "that",
            "talks",
            "began",
            "yesterday",
            "after",
            "the",
            "Q4",
            "2025",
            "audit",
            "and",
            "continue",
            "through",
            "Labor",
            "Day",
            ".",
        ],
    );

    let gold_dates: Vec<GoldSpan> = vec![
        (9, 19, 1, 2, DATE_ENTITY),      // "2026-03-15"
        (44, 58, 6, 10, DATE_ENTITY),    // "March 18, 2026"
        (76, 85, 13, 14, DATE_ENTITY),   // "yesterday"
        (96, 103, 16, 18, DATE_ENTITY),  // "Q4 2025"
        (131, 140, 22, 24, DATE_ENTITY), // "Labor Day"
    ];

    // 1. Deterministic `TemporalExtractor` rule layer finds the 4 structured
    //    calendar dates ("2026-03-15", "March 18, 2026", "yesterday", "Q4 2025").
    let mut temporals = Temporals::new(ViewId::SOURCE);
    temporals.push(TemporalMention {
        span: Span::new(9, 19),
        value: TemporalValue::Day(Date::new(2026, 3, 15).unwrap()),
        origin: Origin::Absolute,
    });
    temporals.push(TemporalMention {
        span: Span::new(44, 58),
        value: TemporalValue::Day(Date::new(2026, 3, 18).unwrap()),
        origin: Origin::Absolute,
    });
    temporals.push(TemporalMention {
        span: Span::new(76, 85),
        value: TemporalValue::Day(Date::new(2026, 3, 14).unwrap()),
        origin: Origin::Relative,
    });
    temporals.push(TemporalMention {
        span: Span::new(96, 103),
        value: TemporalValue::Quarter { year: 2025, quarter: 4 },
        origin: Origin::Absolute,
    });

    // 2. Learned MASC `date` head (#1730):
    //    - Predicts partial span "March 18" (44..52, tokens 6..8) instead of
    //      "March 18, 2026" (boundary truncation error common in BIO heads)
    //    - Finds "yesterday" (76..85, tokens 13..14)
    //    - Misses "2026-03-15" and "Q4 2025"
    //    - Finds named holiday "Labor Day" (131..140, tokens 22..24)
    let masc_kinds = TagSet::new("ner", &["person", "location", "organization", "date"]);
    let mut masc_spans = Spans::new(ViewId::SOURCE);
    masc_spans.push(Span::new(44, 52));
    masc_spans.push(Span::new(76, 85));
    masc_spans.push(Span::new(131, 140));
    let masc_head = Entities {
        spans: masc_spans,
        tokens: vec![IxRange::new(6, 8), IxRange::new(13, 14), IxRange::new(22, 24)],
        kind: vec![Tag(3), Tag(3), Tag(3)],
        kinds: masc_kinds,
        score: vec![0.84, 0.91, 0.89],
    };

    // 3. Peer comparison fixtures (spaCy `en_core_web_trf` and Stanza OntoNotes
    //    `DATE` predictions on the same sentence):
    //    - spaCy `en_core_web_trf` finds all 5 dates (F1 = 1.00)
    //    - Stanza OntoNotes finds 4 of 5 (includes "the Q4 2025" with determiner,
    //      giving 4 TP, 1 FP, 1 FN -> F1 = 0.80)
    let spacy_pred: Vec<GoldSpan> = gold_dates.clone();
    let stanza_pred: Vec<GoldSpan> = vec![
        (9, 19, 1, 2, DATE_ENTITY),
        (44, 58, 6, 10, DATE_ENTITY),
        (76, 85, 13, 14, DATE_ENTITY),
        (92, 103, 15, 18, DATE_ENTITY), // "the Q4 2025"
        (131, 140, 22, 24, DATE_ENTITY),
    ];

    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, toks, Provenance::by("turbo-tokenize:simple")).unwrap();
    doc.add(&Key::DEFAULT, temporals, Provenance::by("turbo-numeric:temporal")).unwrap();
    doc.add(
        &Key::named("masc"),
        masc_head,
        Provenance {
            reads: vec![Key::<Tokens>::DEFAULT.id()],
            model: Some(ModelRef { id: "masc-ne".into(), revision: "r1".into(), sha256: "b".repeat(64) }),
            ..Provenance::by("turbo-tagger:masc-ne")
        },
    )
    .unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Entities>::named("masc"))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(ConsensusEntityMerger::new(
            vec![Key::named("temporal"), Key::named("masc")],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .build()
        .unwrap();

    let res = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(res.iter().all(Result::is_ok), "{res:?}");

    let rule_only = extract_spans(doc.get_key(&Key::<Entities>::named("temporal")).unwrap());
    let masc_only = extract_spans(doc.get_key(&Key::<Entities>::named("masc")).unwrap());
    let merged = extract_spans(doc.get::<Entities>().unwrap());

    let rule_metrics = score_spans(&gold_dates, &rule_only, Some(DATE_ENTITY));
    let masc_metrics = score_spans(&gold_dates, &masc_only, Some(DATE_ENTITY));
    let consensus_metrics = score_spans(&gold_dates, &merged, Some(DATE_ENTITY));
    let spacy_metrics = score_spans(&gold_dates, &spacy_pred, Some(DATE_ENTITY));
    let stanza_metrics = score_spans(&gold_dates, &stanza_pred, Some(DATE_ENTITY));

    // TemporalExtractor rule alone: 4 TP, 0 FP, 1 FN (P = 1.00, R = 0.80, F1 = 0.888...).
    assert_eq!((rule_metrics.tp, rule_metrics.fp, rule_metrics.fn_count), (4, 0, 1));
    assert!((rule_metrics.f1() - 8.0 / 9.0).abs() < 1e-12);

    // MASC head alone: 2 TP ("yesterday", "Labor Day"), 1 FP ("March 18"), 3 FN.
    assert_eq!((masc_metrics.tp, masc_metrics.fp, masc_metrics.fn_count), (2, 1, 3));

    // Consensus (rule priority + MASC fallback): fixes "March 18" -> "March 18, 2026"
    // via rule priority and retains "Labor Day" from MASC fallback: 5 TP, 0 FP, 0 FN (F1 = 1.00).
    assert_eq!((consensus_metrics.tp, consensus_metrics.fp, consensus_metrics.fn_count), (5, 0, 0));
    assert_eq!(consensus_metrics.f1(), 1.0);
    assert!(consensus_metrics.f1() >= rule_metrics.f1());
    assert!(consensus_metrics.f1() > masc_metrics.f1());
    assert_eq!(spacy_metrics.f1(), 1.0);
    assert!((stanza_metrics.f1() - 0.80).abs() < 1e-12);

    let ev = doc.get::<Evidence<Entities>>().unwrap();
    ev.verify(&doc).unwrap();
    assert_eq!(ev.records(), 5);
    // "March 18, 2026" has 2 votes (rule + MASC partial overlap) and carries MASC raw score 0.84.
    assert_eq!(ev.producer_of(1).annotator, "turbo-numeric:temporal");
    assert_eq!(ev.support_of(1).len(), 2);
    assert_eq!(ev.raw_of(1), Some(0.84));
    // "Labor Day" comes from the MASC head with raw score 0.89.
    assert_eq!(ev.producer_of(4).annotator, "turbo-tagger:masc-ne");
    assert_eq!(ev.raw_of(4), Some(0.89));
}
