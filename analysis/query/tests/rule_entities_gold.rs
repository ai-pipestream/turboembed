//! End-to-end gold evaluation of live deterministic `date`, `time`, `money`,
//! and `percent` entity extraction (`turbo-numeric` + `turbo-grammar` +
//! `turbo-tokenize` + `turbo-namefind`).

use turbo_analysis::annotate::Bound;
use turbo_analysis::{Analyzer, Document, Evidence, Ix, Key, Layer};
use turbo_analysis_layers::{DateTime, Entities, NumberNotation, Temporals, Times};
use turbo_grammar::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator, TimeAnnotator};
use turbo_namefind::{
    ConsensusEntityMerger, DATE_ENTITY, EntityMergePolicy, MONEY_ENTITY, PERCENT_ENTITY, RULE_ENTITY_KINDS,
    RuleEntitiesAnnotator, RuleEntityProjector, TIME_ENTITY, TimeEntityFilter,
};
use turbo_numeric::{MoneyExtractor, QuantityExtractor, TemporalExtractor};
use turbo_tokenize::SimpleTokenizer;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Metrics {
    tp: usize,
    fp: usize,
    fn_count: usize,
}

impl Metrics {
    fn add(&mut self, other: Self) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_count += other.fn_count;
    }

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

type GoldMention = (&'static str, &'static str);

struct GoldDoc {
    text: &'static str,
    expected: &'static [GoldMention],
}

const GOLD_CORPUS: &[GoldDoc] = &[
    GoldDoc {
        text: "On 2026-07-14 at 9:30am, Acme Corp closed a $1.2M round, up 15% from Q3 2025.",
        expected: &[
            ("2026-07-14", DATE_ENTITY),
            ("9:30am", TIME_ENTITY),
            ("$1.2M", MONEY_ENTITY),
            ("15%", PERCENT_ENTITY),
            ("Q3 2025", DATE_ENTITY),
        ],
    },
    GoldDoc {
        text: "Report dated July 4, 2026: Yesterday at 14:45 the fund paid EUR 250 and £75.50 after a 4.5 percent fee.",
        expected: &[
            ("July 4, 2026", DATE_ENTITY),
            ("Yesterday", DATE_ENTITY),
            ("14:45", TIME_ENTITY),
            ("EUR 250", MONEY_ENTITY),
            ("£75.50", MONEY_ENTITY),
            ("4.5 percent", PERCENT_ENTITY),
        ],
    },
    GoldDoc {
        text: "Between 14th March 2026 and July 2026, shipments of 80 kg over 2.5 km for 3 hours cost $4,500 at 8% interest.",
        expected: &[
            ("14th March 2026", DATE_ENTITY),
            ("July 2026", DATE_ENTITY),
            ("$4,500", MONEY_ENTITY),
            ("8%", PERCENT_ENTITY),
        ],
    },
    GoldDoc {
        text: "Filing 2026-11-02: Next month the rate drops by 0.75% while the $99.95 rebate expires at 5pm.",
        expected: &[
            ("2026-11-02", DATE_ENTITY),
            ("Next month", DATE_ENTITY),
            ("0.75%", PERCENT_ENTITY),
            ("$99.95", MONEY_ENTITY),
            ("5pm", TIME_ENTITY),
        ],
    },
];

fn predicted_mentions<'a>(text: &'a str, entities: &'a Entities) -> Vec<(&'a str, &'a str)> {
    (0..entities.len())
        .map(|i| {
            let s = entities.spans.get(Ix::new(i as u32));
            let r = entities.tokens[i];
            assert!(r.start < r.end, "entity {i} has empty token range");
            (&text[s.start as usize..s.end as usize], entities.kinds.label(entities.kind[i]))
        })
        .collect()
}

fn evaluate(expected: &[GoldMention], predicted: &[(&str, &str)], kind_filter: Option<&str>) -> Metrics {
    let g: Vec<(&str, &str)> = expected.iter().copied().filter(|x| kind_filter.is_none_or(|k| x.1 == k)).collect();
    let mut p: Vec<(&str, &str)> = predicted.iter().copied().filter(|x| kind_filter.is_none_or(|k| x.1 == k)).collect();
    let mut tp = 0usize;
    for item in &g {
        if let Some(pos) = p.iter().position(|x| x == item) {
            tp += 1;
            p.swap_remove(pos);
        }
    }
    Metrics { tp, fp: p.len(), fn_count: g.len() - tp }
}

#[test]
fn live_rule_pipeline_meets_exact_span_gold_baseline_for_all_four_types() {
    let ref_dt = DateTime::parse("2026-07-14T00:00:00").unwrap();
    let analyzer = Analyzer::builder()
        .add(SimpleTokenizer::default())
        .add(TemporalExtractor::new())
        .add(TimeAnnotator::english().with_reference(ref_dt))
        .add(MoneyExtractor::new(NumberNotation::LatinUs))
        .add(QuantityExtractor::new(NumberNotation::LatinUs))
        .step(RuleEntityProjector::money(Key::DEFAULT, Key::named("money")))
        .step(RuleEntityProjector::quantities(Key::DEFAULT, Key::named("percent")))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(RuleEntityProjector::times_with_filter(
            Key::DEFAULT,
            Key::named("times"),
            TimeEntityFilter::ClockTimesOnly,
        ))
        .step(ConsensusEntityMerger::new(
            vec![Key::named("money"), Key::named("percent"), Key::named("temporal"), Key::named("times")],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .build()
        .unwrap();

    let mut docs: Vec<Document> = GOLD_CORPUS.iter().map(|g| Document::new(g.text)).collect();
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let mut by_kind = [Metrics::default(); 4];
    let mut micro = Metrics::default();

    for (doc, gold_doc) in docs.iter().zip(GOLD_CORPUS) {
        let entities = doc.get::<Entities>().unwrap();
        entities.check().unwrap();
        let ev = doc.get::<Evidence<Entities>>().unwrap();
        ev.verify(doc).unwrap();

        let pred = predicted_mentions(gold_doc.text, entities);
        assert_eq!(pred, gold_doc.expected, "mismatch on document {:?}", gold_doc.text);

        for (k_idx, kind) in RULE_ENTITY_KINDS.into_iter().enumerate() {
            by_kind[k_idx].add(evaluate(gold_doc.expected, &pred, Some(kind)));
        }
        micro.add(evaluate(gold_doc.expected, &pred, None));
    }

    for (k_idx, kind) in RULE_ENTITY_KINDS.into_iter().enumerate() {
        let m = by_kind[k_idx];
        assert!(m.tp >= 3, "expected at least 3 gold mentions for {kind}, got {}", m.tp);
        assert_eq!((m.precision(), m.recall(), m.f1()), (1.0, 1.0, 1.0), "baseline regression on {kind}");
    }
    assert_eq!((micro.tp, micro.fp, micro.fn_count, micro.f1()), (20, 0, 0, 1.0));
}

#[test]
fn grammar_numbers_and_combined_rule_annotator_preserve_provenance() {
    let text = "The contract paid twenty-five dollars and 12.5 percent on 2026-07-14 at 3:30pm.";
    let analyzer = Analyzer::builder()
        .add(SimpleTokenizer::default())
        .add(TemporalExtractor::new())
        .add(TimeAnnotator::english())
        .step(GrammarAnnotator::english().bound())
        .step(RuleEntityProjector::money(GRAMMAR_MONEY, Key::named("money")))
        .step(RuleEntityProjector::quantities(GRAMMAR_PERCENTS, Key::named("percent")))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(RuleEntityProjector::times_with_filter(
            Key::DEFAULT,
            Key::named("times"),
            TimeEntityFilter::ClockTimesOnly,
        ))
        .step(ConsensusEntityMerger::new(
            vec![Key::named("money"), Key::named("percent"), Key::named("temporal"), Key::named("times")],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .step(
            Bound::new(RuleEntitiesAnnotator::with_time_filter(TimeEntityFilter::ClockTimesOnly))
                .read(4, GRAMMAR_MONEY.name)
                .read(5, GRAMMAR_PERCENTS.name)
                .write(0, "combined"),
        )
        .build()
        .unwrap();

    let mut docs = vec![Document::new(text)];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let doc = &docs[0];
    assert!(doc.get::<Temporals>().is_some());
    assert!(doc.get::<Times>().is_some());
    assert!(doc.get_key(&GRAMMAR_MONEY).is_some());
    assert!(doc.get_key(&GRAMMAR_PERCENTS).is_some());

    let merged = doc.get::<Entities>().unwrap();
    let combined = doc.get_key(&Key::<Entities>::named("combined")).unwrap();
    let expected = [
        ("twenty-five dollars", MONEY_ENTITY),
        ("12.5 percent", PERCENT_ENTITY),
        ("2026-07-14", DATE_ENTITY),
        ("3:30pm", TIME_ENTITY),
    ];
    assert_eq!(predicted_mentions(text, merged), expected);
    assert_eq!(predicted_mentions(text, combined), expected);

    let ev = doc.get::<Evidence<Entities>>().unwrap();
    ev.verify(doc).unwrap();
    assert_eq!(ev.records(), 4);
    assert_eq!(ev.producer_of(0).annotator, "turbo-grammar:numbers");
    assert_eq!(ev.producer_of(1).annotator, "turbo-grammar:numbers");
    assert_eq!(ev.producer_of(2).annotator, "turbo-numeric:temporal");
    assert_eq!(ev.producer_of(3).annotator, "turbo-grammar:times");
}
