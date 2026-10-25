use std::marker::PhantomData;

use turbo_analysis::{Ix, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, SparseTerms, Strs, Tokens};
use turbo_chunker::terms::{IndexedTerms, TermStats, TermVectorStack, Vocabulary};

fn vocab(words: &[&str]) -> Vocabulary {
    let mut text = Strs::default();
    for w in words {
        text.push(w);
    }
    Vocabulary { text }
}

fn stats(units: &[&[(u32, u32)]], vocab_len: usize) -> TermStats<Chunks> {
    let mut s =
        TermStats { row: vec![0], term: vec![], tf: vec![], length: vec![], df: vec![0; vocab_len], _of: PhantomData };
    for unit in units {
        let mut len = 0;
        for &(t, tf) in *unit {
            s.term.push(Ix::new(t));
            s.tf.push(tf);
            s.df[t as usize] += 1;
            len += tf;
        }
        s.length.push(len);
        s.row.push(s.term.len() as u32);
    }
    s
}

fn shares_sum_to_one(stack: &TermVectorStack) {
    if stack.total_weight == 0.0 {
        assert!(stack.segments.is_empty() && stack.other_weight == 0.0);
        return;
    }
    let kept: f32 = stack.segments.iter().map(|s| s.share).sum();
    let total = kept + stack.other_weight / stack.total_weight;
    assert!((total - 1.0).abs() < 1e-6, "kept shares plus other fraction = {total}");
    let kept_weight: f32 = stack.segments.iter().map(|s| s.weight).sum();
    assert!((kept_weight + stack.other_weight - stack.total_weight).abs() < 1e-5);
    assert_eq!(stack.segments.len() as u32 + stack.other_term_count, stack.total_terms);
    let kept_freq: u64 = stack.segments.iter().map(|s| s.frequency as u64).sum();
    assert_eq!(kept_freq + stack.other_frequency, stack.total_frequency);
}

#[test]
fn term_stats_stack_ranks_caps_and_accounts() {
    // vocab ids: apple 0, banana 1, cherry 2, date 3
    let v = vocab(&["apple", "banana", "cherry", "date"]);
    let s = stats(&[&[(0, 2), (1, 5), (2, 5), (3, 1)], &[]], 4);
    let stack = TermVectorStack::from_term_stats(&v, &s, Ix::new(0), 2).unwrap();
    // tie at tf 5 broken by ascending term id: banana (1) before cherry (2).
    assert_eq!(
        stack.segments.iter().map(|x| (x.term.as_str(), x.frequency)).collect::<Vec<_>>(),
        vec![("banana", 5), ("cherry", 5)]
    );
    assert_eq!(
        (stack.total_terms, stack.total_frequency, stack.other_term_count, stack.other_frequency),
        (4, 13, 2, 3)
    );
    assert_eq!(stack.total_weight, 13.0);
    assert_eq!(stack.other_weight, 3.0);
    assert!((stack.segments[0].share - 5.0 / 13.0).abs() < 1e-7);
    assert!(stack.segments.iter().all(|x| !x.is_expanded && x.occurrences.is_empty()));
    shares_sum_to_one(&stack);
    // no cap: no other bucket; zero cap: everything is other.
    let all = TermVectorStack::from_term_stats(&v, &s, Ix::new(0), usize::MAX).unwrap();
    assert_eq!((all.segments.len(), all.other_term_count), (4, 0));
    let none = TermVectorStack::from_term_stats(&v, &s, Ix::new(0), 0).unwrap();
    assert_eq!((none.segments.len(), none.other_term_count, none.other_frequency), (0, 4, 13));
    shares_sum_to_one(&all);
    shares_sum_to_one(&none);
    // the empty second unit stacks to zeros.
    let empty = TermVectorStack::from_term_stats(&v, &s, Ix::new(1), 3).unwrap();
    assert_eq!((empty.total_terms, empty.total_weight), (0, 0.0));
    shares_sum_to_one(&empty);
}

#[test]
fn indexed_terms_stack_counts_the_document_with_occurrences() {
    let v = vocab(&["apple", "banana", "cherry"]);
    let indexed = IndexedTerms {
        term: vec![Some(Ix::new(1)), None, Some(Ix::new(0)), Some(Ix::new(1)), Some(Ix::new(1)), Some(Ix::new(2))],
    };
    let mut spans = Spans::<Tokens>::new(ViewId(0));
    for i in 0..6u32 {
        spans.push(Span { start: i * 2, end: i * 2 + 1 });
    }
    let tokens = Tokens { spans, sentence: None };
    let stack = TermVectorStack::from_indexed_terms(&v, &indexed, Some(&tokens), 2).unwrap();
    assert_eq!(stack.segments[0].term, "banana");
    assert_eq!(stack.segments[0].frequency, 3);
    assert_eq!(
        stack.segments[0].occurrences,
        vec![Span { start: 0, end: 1 }, Span { start: 6, end: 7 }, Span { start: 8, end: 9 }],
        "occurrences are the carrying tokens in order"
    );
    // apple (id 0) edges cherry (id 2) on the tf-1 tie.
    assert_eq!(stack.segments[1].term, "apple");
    assert_eq!(stack.other_term_count, 1);
    shares_sum_to_one(&stack);
    // without tokens: same ranking, no occurrences.
    let plain = TermVectorStack::from_indexed_terms(&v, &indexed, None, 3).unwrap();
    assert!(plain.segments.iter().all(|x| x.occurrences.is_empty()));
    assert_eq!(plain.total_frequency, 5);
}

#[test]
fn sparse_stack_weights_expansion_and_occurrences() {
    let mut sparse = SparseTerms::<Chunks>::new(ModelRef { id: "m".into(), revision: "t".into(), sha256: "s".into() });
    // unit 0: term 7 (two occurrences, w 1.5), term 3 (expanded, w 2.0), term 9 (one occurrence, w 1.5)
    sparse.push(3, 2.0, None);
    sparse.push_occurrences(&[]);
    sparse.push(7, 1.5, Some(Span { start: 0, end: 9 }));
    sparse.push_occurrences(&[Span { start: 0, end: 3 }, Span { start: 6, end: 9 }]);
    sparse.push(9, 1.5, Some(Span { start: 10, end: 12 }));
    sparse.push_occurrences(&[Span { start: 10, end: 12 }]);
    sparse.unit();
    let names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let stack =
        TermVectorStack::from_sparse_terms(&sparse, Ix::new(0), |id| names.get(id as usize).copied(), 10).unwrap();
    // weight desc, then frequency desc: three (2.0) first; at 1.5, seven (2 occ) before nine (1 occ).
    assert_eq!(
        stack.segments.iter().map(|x| (x.term.as_str(), x.frequency, x.is_expanded)).collect::<Vec<_>>(),
        vec![("three", 0, true), ("seven", 2, false), ("nine", 1, false)]
    );
    assert_eq!(stack.segments[1].occurrences.len(), 2);
    assert_eq!(stack.total_weight, 5.0);
    shares_sum_to_one(&stack);
    // no occurrence table: the entry span is the occurrence.
    let mut sparse = SparseTerms::<Chunks>::new(ModelRef { id: "m".into(), revision: "t".into(), sha256: "s".into() });
    sparse.push(1, 0.5, Some(Span { start: 2, end: 4 }));
    sparse.push(2, 0.25, None);
    sparse.unit();
    let stack =
        TermVectorStack::from_sparse_terms(&sparse, Ix::new(0), |id| names.get(id as usize).copied(), 10).unwrap();
    assert_eq!(stack.segments[0].occurrences, vec![Span { start: 2, end: 4 }]);
    assert!(!stack.segments[0].is_expanded && stack.segments[1].is_expanded);
}

#[test]
fn stack_refusals() {
    let v = vocab(&["apple", "banana"]);
    let s = stats(&[&[(0, 1)]], 2);
    // missing unit
    assert!(TermVectorStack::from_term_stats(&v, &s, Ix::new(7), 3).unwrap_err().contains("unit 7"));
    // term id outside the vocabulary (df sized to a larger vocab)
    let wide = stats(&[&[(5, 1)]], 6);
    assert!(TermVectorStack::from_term_stats(&v, &wide, Ix::new(0), 3).unwrap_err().contains("term 5"));
    // invalid stats (tf column out of step)
    let mut broken = stats(&[&[(0, 1)]], 2);
    broken.tf.clear();
    assert!(TermVectorStack::from_term_stats(&v, &broken, Ix::new(0), 3).is_err());
    // unresolvable sparse id
    let mut sparse = SparseTerms::<Chunks>::new(ModelRef { id: "m".into(), revision: "t".into(), sha256: "s".into() });
    sparse.push(99, 1.0, None);
    sparse.unit();
    assert!(TermVectorStack::from_sparse_terms(&sparse, Ix::new(0), |_| None, 3).unwrap_err().contains("99"));
    // tokens misaligned with indexed terms
    let indexed = IndexedTerms { term: vec![Some(Ix::new(0))] };
    let tokens = Tokens { spans: Spans::<Tokens>::new(ViewId(0)), sentence: None };
    assert!(TermVectorStack::from_indexed_terms(&v, &indexed, Some(&tokens), 3).is_err());
}
