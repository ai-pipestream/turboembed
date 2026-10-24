//! The default pattern libraries measured on the in-tree evaluation
//! fixture (`tests/fixtures/defaults/eval.conllu`): gold UD parses,
//! gold entities and gold (head, tail, label) triples. Each library is
//! scored against the gold triples of its own labels, a predicted
//! triple counting once however many patterns produced it. The numbers
//! asserted here are the ones docs/relation.md records.

use std::collections::{BTreeMap, BTreeSet};

use turbo_analysis::{Annotator, Document, Ix, IxRange, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Strs, Tokens};
use turbo_relation::RelationAnnotator;
use turbo_relation::defaults::{english_defaults, english_semantic_defaults};

const FIXTURE: &str = include_str!("fixtures/defaults/eval.conllu");

/// One gold sentence: its words (form, lemma, head, label), entities
/// (first and last word, 0-based and inclusive, and kind) and gold
/// relations (head and tail entity, 0-based, and label).
#[derive(Default)]
struct Sentence {
    words: Vec<(String, String, usize, String)>,
    entities: Vec<(usize, usize, String)>,
    gold: Vec<(usize, usize, String)>,
}

fn number(s: &str) -> usize {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn parse() -> Vec<Sentence> {
    let mut out = Vec::new();
    let mut cur = Sentence::default();
    for line in FIXTURE.lines().chain(std::iter::once("")) {
        if line.is_empty() {
            if !cur.words.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else if let Some(e) = line.strip_prefix("# entity = ") {
            let (range, kind) = e.split_once(' ').expect("entity: range and kind");
            let (a, b) = range.split_once('-').expect("entity: first-last");
            cur.entities.push((number(a) - 1, number(b) - 1, kind.to_owned()));
        } else if let Some(r) = line.strip_prefix("# relation = ") {
            let f: Vec<&str> = r.split(' ').collect();
            assert_eq!(f.len(), 3, "{line}");
            cur.gold.push((number(f[0]) - 1, number(f[1]) - 1, f[2].to_owned()));
        } else if !line.starts_with('#') {
            let c: Vec<&str> = line.split('\t').collect();
            assert_eq!(c.len(), 10, "{line}");
            assert_eq!(number(c[0]), cur.words.len() + 1, "{line}");
            cur.words.push((c[1].to_owned(), c[2].to_owned(), number(c[6]), c[7].to_owned()));
        }
    }
    out
}

/// The sentence's (head span, tail span, label) triples, as source
/// character ranges, from the libraries' relations.
type Triple = (u32, u32, u32, u32, String);

fn extract(s: &Sentence, annotator: &RelationAnnotator) -> (BTreeSet<Triple>, BTreeSet<Triple>) {
    let text = s.words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(" ");
    let mut starts = Vec::new();
    let mut at = 0u32;
    for w in &s.words {
        starts.push(at);
        at += w.0.len() as u32 + 1;
    }
    let end = |i: usize| starts[i] + s.words[i].0.len() as u32;
    let mut token_spans = Spans::new(ViewId::SOURCE);
    for (i, &start) in starts.iter().enumerate() {
        token_spans.push(Span::new(start, end(i)));
    }
    let n = s.words.len();
    let tokens = Tokens { spans: token_spans, sentence: Some(vec![Ix::new(0); n]) };
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    sentence_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: sentence_spans };
    let mut labels: Vec<&str> = Vec::new();
    for w in &s.words {
        if !labels.contains(&w.3.as_str()) {
            labels.push(&w.3);
        }
    }
    let deps = Dependencies {
        head: s.words.iter().map(|w| w.2.checked_sub(1).map(|h| Ix::new(h as u32))).collect(),
        relation: s.words.iter().map(|w| Tag(labels.iter().position(|l| *l == w.3).unwrap() as u16)).collect(),
        relations: TagSet::new("dependency", &labels),
    };
    let mut kinds: Vec<&str> = Vec::new();
    for e in &s.entities {
        if !kinds.contains(&e.2.as_str()) {
            kinds.push(&e.2);
        }
    }
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    for e in &s.entities {
        entity_spans.push(Span::new(starts[e.0], end(e.1)));
    }
    let entities = Entities {
        spans: entity_spans,
        tokens: s.entities.iter().map(|e| IxRange::new(e.0 as u32, e.1 as u32 + 1)).collect(),
        kind: s.entities.iter().map(|e| Tag(kinds.iter().position(|k| *k == e.2).unwrap() as u16)).collect(),
        kinds: TagSet::new("entity", &kinds),
        score: vec![1.0; s.entities.len()],
    };
    let mut lemma = Strs::default();
    for w in &s.words {
        lemma.push(&w.1);
    }
    let lemmas = Lemmas { lemma };
    let doc = Document::new(text.clone());
    let view = doc.get::<View>().unwrap();
    let (relations, _) = annotator
        .annotate((view, &sentences, &tokens, &tokens, &deps, &entities, None::<&PosTags>, Some(&lemmas)))
        .unwrap();
    let span = |e: usize| (starts[s.entities[e].0], end(s.entities[e].1));
    let triple = |h: usize, t: usize, label: &str| {
        let ((a, b), (c, d)) = (span(h), span(t));
        (a, b, c, d, label.to_owned())
    };
    let mut predicted = BTreeSet::new();
    for i in 0..relations.spans.len() {
        let (h, t, label, _) = relations.mention(Ix::new(i as u32));
        predicted.insert(triple(h.get(), t.get(), label));
    }
    let gold = s.gold.iter().map(|g| triple(g.0, g.1, &g.2)).collect();
    (predicted, gold)
}

/// Per label: true positives, predicted, gold; for the labels of
/// `library`, over the whole fixture.
fn score(sentences: &[Sentence], library: Vec<turbo_relation::RelationPattern>) -> BTreeMap<String, [u32; 3]> {
    let labels: BTreeSet<String> = library.iter().map(|p| p.relation_type().to_owned()).collect();
    let annotator = RelationAnnotator::new(library).unwrap();
    let mut out: BTreeMap<String, [u32; 3]> = labels.iter().map(|l| (l.clone(), [0; 3])).collect();
    for s in sentences {
        let (predicted, gold) = extract(s, &annotator);
        for t in &predicted {
            let c = out.get_mut(&t.4).expect("a library label");
            c[1] += 1;
            c[0] += u32::from(gold.contains(t));
        }
        for t in gold.iter().filter(|t| labels.contains(&t.4)) {
            out.get_mut(&t.4).unwrap()[2] += 1;
        }
    }
    out
}

/// Precision, recall and F1 to three places; `-` where undefined.
fn prf(c: [u32; 3]) -> String {
    let ratio = |a: u32, b: u32| if b == 0 { None } else { Some(f64::from(a) / f64::from(b)) };
    let (p, r) = (ratio(c[0], c[1]), ratio(c[0], c[2]));
    let f = match (p, r) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (_, Some(_)) => Some(0.0),
        _ => None,
    };
    let show = |x: Option<f64>| x.map_or("-".to_owned(), |x| format!("{x:.3}"));
    format!("{} {} {}", show(p), show(r), show(f))
}

fn table(scores: &BTreeMap<String, [u32; 3]>) -> Vec<String> {
    let mut rows: Vec<String> =
        scores.iter().map(|(l, c)| format!("{l} {}/{}/{} {}", c[0], c[1], c[2], prf(*c))).collect();
    let mut all = [0u32; 3];
    for c in scores.values() {
        for i in 0..3 {
            all[i] += c[i];
        }
    }
    rows.push(format!("overall {}/{}/{} {}", all[0], all[1], all[2], prf(all)));
    rows
}

#[test]
fn the_fixture_is_well_formed() {
    let sentences = parse();
    assert_eq!(sentences.len(), 26);
    for (i, s) in sentences.iter().enumerate() {
        assert_eq!(s.words.iter().filter(|w| w.2 == 0).count(), 1, "sentence {}: one root", i + 1);
        assert!(s.words.iter().all(|w| w.2 <= s.words.len()), "sentence {}: a head past the end", i + 1);
        for e in &s.entities {
            assert!(e.0 <= e.1 && e.1 < s.words.len(), "sentence {}: entity {e:?}", i + 1);
        }
        for g in &s.gold {
            assert!(g.0 < s.entities.len() && g.1 < s.entities.len() && g.0 != g.1, "sentence {}: {g:?}", i + 1);
        }
    }
    let words: usize = sentences.iter().map(|s| s.words.len()).sum();
    let entities: usize = sentences.iter().map(|s| s.entities.len()).sum();
    let gold: usize = sentences.iter().map(|s| s.gold.len()).sum();
    assert_eq!((words, entities, gold), (187, 61, 39));
}

#[test]
fn the_shape_defaults_score_as_recorded() {
    let got = table(&score(&parse(), english_defaults()));
    let want = [
        "apposition 2/2/2 1.000 1.000 1.000",
        "attribute 2/4/2 0.500 1.000 0.667",
        "passive-agent 3/8/3 0.375 1.000 0.545",
        "possession 2/2/2 1.000 1.000 1.000",
        "prepositional-attachment 3/3/3 1.000 1.000 1.000",
        "subject-object 6/6/6 1.000 1.000 1.000",
        "overall 18/25/18 0.720 1.000 0.837",
    ];
    assert_eq!(got, want);
}

#[test]
fn the_semantic_defaults_score_as_recorded() {
    let got = table(&score(&parse(), english_semantic_defaults()));
    let want = [
        "acquired_by 2/2/3 1.000 0.667 0.800",
        "born_in 1/1/2 1.000 0.500 0.667",
        "founded_by 1/1/2 1.000 0.500 0.667",
        "headquartered_in 1/1/1 1.000 1.000 1.000",
        "located_in 0/0/4 - 0.000 0.000",
        "member_of 0/1/2 0.000 0.000 0.000",
        "spouse_of 0/0/2 - 0.000 0.000",
        "works_for 1/2/5 0.500 0.200 0.286",
        "overall 6/8/21 0.750 0.286 0.414",
    ];
    assert_eq!(got, want);
}
