//! A small salience evaluation: eight hand-labelled news-style documents,
//! each with one gold central entity, scored by top-1 accuracy and mean
//! reciprocal rank (MRR) of the gold entity for `EntitySalienceAnnotator`
//! against two baselines, first mention and mention count. The mentions
//! and their head roles are labelled by hand (no tagger, parser or
//! consolidator in the loop), so this is a regression check on the
//! scoring, not a measure of end-to-end quality. Two cases are known
//! misses, kept as written rather than tuned away: a topic named in one
//! section heading and then mostly as an object (5), and a topic that is
//! mostly an object while a regulator is the repeated subject (7).

mod salience_common;

use salience_common::{M, O, S, Sentence, X, build, ranked};
use turbo_analysis::{Document, Key};
use turbo_analysis_layers::DocumentEntities;
use turbo_entity_linker::EntitySalienceAnnotator;

struct Case {
    sentences: &'static [Sentence],
    heading: Option<usize>,
    gold: &'static str,
}

const CASES: &[Case] = &[
    // A dateline wire service comes first; the company is the topic.
    Case {
        sentences: &[
            &[("Reuters", S), ("Acme", S)],
            &[("Acme", S), ("Ohio", X)],
            &[("Acme", S)],
            &[("Union", S), ("Acme", O)],
        ],
        heading: None,
        gold: "Acme",
    },
    // A city repeated as a modifier outnumbers the subject.
    Case {
        sentences: &[
            &[("Ortiz", S), ("Boston", M)],
            &[("Boston", M), ("Ortiz", S)],
            &[("Boston", M)],
            &[("Boston", M), ("Ortiz", S)],
            &[("Boston", X)],
        ],
        heading: None,
        gold: "Ortiz",
    },
    // A place leads the story; the company is the repeated subject.
    Case {
        sentences: &[
            &[("Paris", X), ("Mistral", S)],
            &[("Mistral", S), ("Nvidia", O)],
            &[("Mistral", S)],
            &[("Nvidia", S)],
            &[("Mistral", S)],
        ],
        heading: None,
        gold: "Mistral",
    },
    // Easy: the lead subject is also the most frequent.
    Case {
        sentences: &[
            &[("Tesla", S), ("Musk", M)],
            &[("Tesla", S)],
            &[("Musk", S), ("Tesla", O)],
            &[("Tesla", S), ("Texas", X)],
        ],
        heading: None,
        gold: "Tesla",
    },
    // A section heading names the topic, mostly an object in the body. Miss.
    Case {
        sentences: &[
            &[("Sweden", S)],
            &[("Volvo", X)],
            &[("Sweden", S), ("Volvo", O)],
            &[("Volvo", S)],
            &[("Sweden", M)],
        ],
        heading: Some(1),
        gold: "Volvo",
    },
    // A hub entity co-occurs with every other one.
    Case {
        sentences: &[
            &[("Senate", S), ("Bill", O)],
            &[("House", S), ("Bill", O)],
            &[("Bill", S), ("Biden", O)],
            &[("Senate", S)],
            &[("House", S)],
        ],
        heading: None,
        gold: "Bill",
    },
    // The topic is mostly an object; the regulator is the subject. Miss.
    Case {
        sentences: &[
            &[("Commission", S), ("BetaBank", O)],
            &[("Commission", S), ("BetaBank", O)],
            &[("BetaBank", S)],
            &[("Commission", S), ("BetaBank", O)],
        ],
        heading: None,
        gold: "BetaBank",
    },
    // Two equals; the first subject is gold by the headline convention.
    Case {
        sentences: &[&[("Lakers", S), ("Celtics", O)], &[("Celtics", S)], &[("Lakers", S)], &[("Celtics", X)]],
        heading: None,
        gold: "Lakers",
    },
];

/// The canonical names in order of first mention, then by mention count
/// (stable, so ties keep first-mention order).
fn baselines(d: &Document) -> (Vec<String>, Vec<String>) {
    let g = d.require(&Key::<DocumentEntities>::DEFAULT).unwrap();
    let mut first: Vec<(u32, usize, String)> =
        g.entities.iter().map(|e| (e.mentions[0].span.start, e.mentions.len(), e.canonical_name.to_string())).collect();
    first.sort_by_key(|r| r.0);
    let mut count = first.clone();
    count.sort_by_key(|r| std::cmp::Reverse(r.1));
    (first.into_iter().map(|r| r.2).collect(), count.into_iter().map(|r| r.2).collect())
}

/// (top-1 accuracy, MRR) of the gold entity over the cases.
fn score(rankings: &[(Vec<String>, &str)]) -> (f64, f64) {
    let n = rankings.len() as f64;
    let (mut hit, mut rr) = (0.0, 0.0);
    for (r, gold) in rankings {
        let at = r.iter().position(|x| x == gold).expect("gold present");
        hit += f64::from(at == 0);
        rr += 1.0 / (at as f64 + 1.0);
    }
    (hit / n, rr / n)
}

#[test]
fn salience_beats_first_mention_and_frequency() {
    let (mut sal, mut first, mut freq) = (Vec::new(), Vec::new(), Vec::new());
    for c in CASES {
        let d = build(c.sentences, c.heading, true);
        let s = EntitySalienceAnnotator::new().salience(&d).unwrap();
        let (f, q) = baselines(&d);
        sal.push((ranked(&d, &s), c.gold));
        first.push((f, c.gold));
        freq.push((q, c.gold));
    }
    let misses: Vec<&str> = sal.iter().filter(|(r, g)| r[0] != *g).map(|(_, g)| *g).collect();
    let (s, f, q) = (score(&sal), score(&first), score(&freq));
    println!("salience      top1 {:.3} mrr {:.3}", s.0, s.1);
    println!("first mention top1 {:.3} mrr {:.3}", f.0, f.1);
    println!("frequency     top1 {:.3} mrr {:.3}", q.0, q.1);
    println!("salience misses: {misses:?}");
    assert_eq!(misses, ["Volvo", "BetaBank"]);
    assert!(s.0 > f.0 && s.0 > q.0, "{s:?} {f:?} {q:?}");
    assert!(s.1 > f.1 && s.1 > q.1, "{s:?} {f:?} {q:?}");
}
