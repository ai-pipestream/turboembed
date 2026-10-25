//! A small hand-built evaluation of `EventExtractor`, not a corpus score:
//! ten news and finance sentences with hand-written UD parses and gold
//! `(trigger, role, argument)` triples annotated under the role
//! definitions of `ArgumentRole`. The parses are gold, so this measures
//! the role mapping and the anchoring, not parsing. Three systems are
//! scored by exact match: subject and object only (what a binary
//! relation triple keeps), every role without anchoring, and the full
//! extractor.

mod event_support;

use std::collections::BTreeSet;

use event_support::*;
use turbo_analysis::Document;
use turbo_analysis_layers::{Date, TemporalValue};
use turbo_relation::EventConfig;

type Gold = &'static [(&'static str, &'static str, &'static str)];

fn corpus() -> Vec<(Document, Gold)> {
    let mut out = Vec::new();
    let mut d = parsed(
        "Shares fell 3% on Tuesday after the earnings report .",
        "2:nsubj 0:root 2:obl:npmod 5:case 2:obl 9:case 9:det 9:compound 2:obl 2:punct",
    );
    add_quantities(&mut d, &[(2, 3)]);
    add_temporals(&mut d, &[((4, 5), TemporalValue::Day(Date::new(2026, 10, 20).unwrap()))]);
    out.push((
        d,
        &[
            ("fell", "Agent", "Shares"),
            ("fell", "Measure", "3%"),
            ("fell", "Time", "Tuesday"),
            ("fell", "Time", "the earnings report"),
        ] as Gold,
    ));

    let mut d = parsed(
        "The board approved a $2B buyback in May .",
        "2:det 3:nsubj 0:root 6:det 6:compound 3:obj 8:case 3:obl 3:punct",
    );
    add_money(&mut d, &[(4, 5)]);
    add_temporals(&mut d, &[((7, 8), TemporalValue::Month { year: 2026, month: 5 })]);
    out.push((
        d,
        &[("approved", "Agent", "The board"), ("approved", "Patient", "a $2B buyback"), ("approved", "Time", "May")],
    ));

    let mut d = parsed(
        "Initech was fined $5M by regulators in Brussels .",
        "3:nsubj:pass 3:aux:pass 0:root 3:obj 6:case 3:obl 8:case 3:obl 3:punct",
    );
    add_money(&mut d, &[(3, 4)]);
    add_entities(&mut d, &[((0, 1), "organization"), ((7, 8), "location")]);
    out.push((
        d,
        &[
            ("fined", "Patient", "Initech"),
            ("fined", "Measure", "$5M"),
            ("fined", "Agent", "regulators"),
            ("fined", "Location", "Brussels"),
        ],
    ));

    let mut d = parsed(
        "Sales grew 8% and margins widened to 12% .",
        "2:nsubj 0:root 2:obj 6:cc 6:nsubj 2:conj 8:case 6:obl 2:punct",
    );
    add_quantities(&mut d, &[(2, 3), (7, 8)]);
    out.push((
        d,
        &[
            ("grew", "Agent", "Sales"),
            ("grew", "Measure", "8%"),
            ("widened", "Agent", "margins"),
            ("widened", "Measure", "12%"),
        ],
    ));

    let mut d = parsed(
        "Acme hired 200 engineers in Denver to expand its cloud unit .",
        "2:nsubj 0:root 4:nummod 2:obj 6:case 2:obl 8:mark 2:advcl 11:nmod:poss 11:compound 8:obj 2:punct",
    );
    add_entities(&mut d, &[((0, 1), "organization"), ((5, 6), "location")]);
    out.push((
        d,
        &[
            ("hired", "Agent", "Acme"),
            ("hired", "Patient", "200 engineers"),
            ("hired", "Location", "Denver"),
            ("hired", "Purpose", "expand its cloud unit"),
            ("expand", "Agent", "Acme"),
            ("expand", "Patient", "its cloud unit"),
        ],
    ));

    let d =
        parsed("Prices rose because demand outpaced supply .", "2:nsubj 0:root 5:mark 5:nsubj 2:advcl 5:obj 2:punct");
    out.push((
        d,
        &[
            ("rose", "Agent", "Prices"),
            ("rose", "Cause", "demand outpaced supply"),
            ("outpaced", "Agent", "demand"),
            ("outpaced", "Patient", "supply"),
        ],
    ));

    let mut d = parsed(
        "The merger was announced on March 3 , 2026 and closed in June .",
        "2:det 4:nsubj:pass 4:aux:pass 0:root 6:case 4:obl 6:nummod 9:punct 6:nummod 11:cc 4:conj 13:case 11:obl 4:punct",
    );
    add_temporals(
        &mut d,
        &[
            ((5, 9), TemporalValue::Day(Date::new(2026, 3, 3).unwrap())),
            ((12, 13), TemporalValue::Month { year: 2026, month: 6 }),
        ],
    );
    out.push((
        d,
        &[
            ("announced", "Patient", "The merger"),
            ("announced", "Time", "March 3 , 2026"),
            ("closed", "Agent", "The merger"),
            ("closed", "Time", "June"),
        ],
    ));

    let mut d = parsed(
        "Globex paid shareholders a dividend of $1.20 per share .",
        "2:nsubj 0:root 2:iobj 5:det 2:obj 7:case 5:nmod 9:case 7:nmod 2:punct",
    );
    add_money(&mut d, &[(6, 7)]);
    out.push((
        d,
        &[
            ("paid", "Agent", "Globex"),
            ("paid", "Recipient", "shareholders"),
            ("paid", "Patient", "a dividend of $1.20 per share"),
            ("paid", "Measure", "$1.20"),
        ],
    ));

    let mut d = parsed(
        "Analysts expect revenue of $3B next year .",
        "2:nsubj 0:root 2:obj 5:case 3:nmod 7:amod 2:obl:tmod 2:punct",
    );
    add_money(&mut d, &[(4, 5)]);
    add_temporals(&mut d, &[((5, 7), TemporalValue::Year(2027))]);
    out.push((
        d,
        &[("expect", "Agent", "Analysts"), ("expect", "Patient", "revenue of $3B"), ("expect", "Time", "next year")],
    ));

    let mut d = parsed(
        "The plant in Ohio was closed with little warning .",
        "2:det 6:nsubj:pass 4:case 2:nmod 6:aux:pass 0:root 9:case 9:amod 6:obl 6:punct",
    );
    add_entities(&mut d, &[((3, 4), "location")]);
    out.push((d, &[("closed", "Patient", "The plant in Ohio"), ("closed", "Location", "Ohio")]));
    out
}

struct Score {
    tp: usize,
    predicted: usize,
    gold: usize,
}

impl Score {
    fn f1(&self) -> (f64, f64, f64) {
        let p = self.tp as f64 / self.predicted.max(1) as f64;
        let r = self.tp as f64 / self.gold.max(1) as f64;
        (p, r, if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) })
    }
}

fn score(config: EventConfig, roles: Option<&[&str]>, only: Option<&[&str]>) -> Score {
    let mut s = Score { tp: 0, predicted: 0, gold: 0 };
    for (d, gold) in corpus() {
        let keep = |role: &str| only.is_none_or(|o| o.contains(&role));
        let found: BTreeSet<(String, String, String)> = rows(&d, &events(&d, config.clone()))
            .into_iter()
            .filter(|r| roles.is_none_or(|k| k.contains(&r.1.as_str())) && keep(&r.1))
            .map(|r| (r.0, r.1, r.2))
            .collect();
        let gold: BTreeSet<(String, String, String)> =
            gold.iter().filter(|g| keep(g.1)).map(|&(t, r, a)| (t.into(), r.into(), a.into())).collect();
        for miss in gold.difference(&found) {
            println!("missed {miss:?}");
        }
        for extra in found.difference(&gold) {
            println!("extra  {extra:?}");
        }
        s.tp += found.intersection(&gold).count();
        s.predicted += found.len();
        s.gold += gold.len();
    }
    s
}

#[test]
fn anchoring_lifts_argument_f1_over_triples() {
    let plain = EventConfig { anchor: false, ..Default::default() };
    let systems = [
        ("subject and object only", score(plain.clone(), Some(&["Agent", "Patient"]), None)),
        ("all roles, no anchoring", score(plain.clone(), None, None)),
        ("full extractor", score(EventConfig::default(), None, None)),
    ];
    for (name, s) in &systems {
        let (p, r, f) = s.f1();
        println!("{name}: {}/{} gold, {} predicted, P {p:.3} R {r:.3} F1 {f:.3}", s.tp, s.gold, s.predicted);
    }
    let anchored = ["Time", "Measure", "Location"];
    for (name, config) in [("no anchoring", plain), ("full", EventConfig::default())] {
        let s = score(config, None, Some(&anchored));
        let (p, r, f) = s.f1();
        println!("time, measure, location, {name}: {}/{} P {p:.3} R {r:.3} F1 {f:.3}", s.tp, s.gold);
    }
    let f1: Vec<f64> = systems.iter().map(|(_, s)| s.f1().2).collect();
    assert!(f1[2] > f1[1] && f1[1] > f1[0], "{f1:?}");
    assert!(f1[2] >= 0.85, "{f1:?}");
}
