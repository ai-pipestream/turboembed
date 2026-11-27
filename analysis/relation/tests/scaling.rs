//! Deterministic scaling checks for [`RelationAnnotator`] and
//! [`RelationPathAnnotator`].
//!
//! Two non-clock measures guard against superlinear regressions without
//! sensitivity to shared CI runner load:
//!
//! - [`RelationWork`] counters (`head_probes`, `pair_visits`, `arc_reads`,
//!   `edge_ops`) recorded at the input probes, pair loop, parent-arc reads,
//!   and edge merges.
//! - Allocation call counts and byte totals recorded by a thread-scoped
//!   counting allocator.
//!
//! Run `wall_clock_timing_probe` with `--ignored --nocapture` for manual
//! wall-clock timing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use turbo_analysis::{Annotator, Document, Edges, Ix, IxRange, Layer, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::{RelationAnnotator, RelationPathAnnotator, RelationPattern, RelationWork, Relations};

struct Counting;

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count_alloc(bytes: usize) {
    let _ = CALLS.try_with(|c| c.set(c.get() + 1));
    let _ = BYTES.try_with(|b| b.set(b.get() + bytes));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_alloc(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_alloc(size);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Layers over `text`: one token per `(start, end, head, label)`, one
/// sentence per range, and one entity per listed token.
struct Input {
    doc: Document,
    sentences: Sentences,
    tokens: Tokens,
    deps: Dependencies,
    entities: Entities,
}

fn input(
    text: &str,
    sentences: &[(u32, u32)],
    tokens: &[(u32, u32, Option<u32>, u16)],
    entity_tokens: &[u32],
) -> Input {
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in sentences {
        sentence_spans.push(Span::new(a, b));
    }
    let mut token_spans = Spans::new(ViewId::SOURCE);
    let mut sentence = Vec::new();
    for &(a, b, _, _) in tokens {
        token_spans.push(Span::new(a, b));
        sentence.push(Ix::new(sentences.partition_point(|&(s, _)| s <= a) as u32 - 1));
    }
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    for &t in entity_tokens {
        let (a, b, _, _) = tokens[t as usize];
        entity_spans.push(Span::new(a, b));
    }
    let n = entity_tokens.len();
    Input {
        doc: Document::new(text),
        sentences: Sentences { spans: sentence_spans },
        tokens: Tokens { spans: token_spans, sentence: Some(sentence) },
        deps: Dependencies {
            head: tokens.iter().map(|t| t.2.map(Ix::new)).collect(),
            relation: tokens.iter().map(|t| Tag(t.3)).collect(),
            relations: TagSet::new("dependency", &["root", "conj", "nsubj", "obj", "ccomp"]),
        },
        entities: Entities {
            spans: entity_spans,
            tokens: entity_tokens.iter().map(|&t| IxRange::new(t, t + 1)).collect(),
            kind: vec![Tag(0); n],
            kinds: TagSet::new("entity", &["item"]),
            score: vec![1.0; n],
        },
    }
}

struct Sample {
    relations: Relations,
    edges: Edges<Entities, Entities>,
    work: RelationWork,
    alloc_calls: usize,
    alloc_bytes: usize,
}

fn run_counted(annotator: &RelationAnnotator, inp: &Input) -> Sample {
    let view = inp.doc.get::<View>().unwrap();
    let (c0, b0) = (CALLS.get(), BYTES.get());
    let ((relations, edges), work) = annotator
        .annotate_with_work((view, &inp.sentences, &inp.tokens, &inp.tokens, &inp.deps, &inp.entities, None, None))
        .unwrap();
    let (alloc_calls, alloc_bytes) = (CALLS.get() - c0, BYTES.get() - b0);
    Sample { relations, edges, work, alloc_calls, alloc_bytes }
}

fn assert_bytes_scale_linearly(label: &str, samples: &[(usize, usize)]) {
    for pair in samples.windows(2) {
        let ((u_prev, b_prev), (u_curr, b_curr)) = (pair[0], pair[1]);
        let lhs = (b_curr as u128) * (u_prev as u128) * 100;
        let rhs = (b_prev as u128) * (u_curr as u128) * 125;
        assert!(
            lhs <= rhs,
            "{label}: allocated bytes grew faster than linear ({u_prev} -> {u_curr} units): {b_prev} -> {b_curr}"
        );
    }
}

/// One head with `n` `conj` dependents, all entities: a sibling pattern
/// matches all `n(n - 1)` ordered pairs.
fn coordination(n: u32) -> Input {
    let text: String = std::iter::once("h").chain(std::iter::repeat_n(" e", n as usize)).collect();
    let mut tokens = vec![(0, 1, None, 0)];
    tokens.extend((0..n).map(|i| (2 + 2 * i, 3 + 2 * i, Some(0), 1)));
    input(&text, &[(0, text.len() as u32)], &tokens, &(1..=n).collect::<Vec<_>>())
}

#[test]
fn edges_grow_linearly_with_mentions() {
    let annotator: RelationAnnotator = RelationAnnotator::new(vec![
        RelationPattern::new("r", "<conj >conj").unwrap().confidence(0.5).unwrap(),
        RelationPattern::new("r", "<conj >conj").unwrap().confidence(0.9).unwrap(),
    ])
    .unwrap();
    let sizes = [75u32, 150, 300];
    let mut work_samples = Vec::new();
    let mut byte_samples = Vec::new();
    let mut last = None;
    for n in sizes {
        let mentions = (n as usize) * (n as usize - 1);
        let s = run_counted(&annotator, &coordination(n));
        assert_eq!(s.relations.subject.len(), 2 * mentions);
        assert_eq!(s.edges.len(), mentions);
        // Zero per-pair heap allocations: 89,700 pairs run inside a fixed set of vector growths.
        assert!(s.alloc_calls < 256, "n={n}: expected < 256 heap allocations, got {}", s.alloc_calls);
        work_samples.push((mentions, s.work));
        byte_samples.push((mentions, s.alloc_bytes));
        last = Some(s);
    }
    RelationWork::check_linear_scaling(&work_samples).unwrap();
    assert_bytes_scale_linearly("coordination", &byte_samples);

    let single: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("r", "<conj >conj").unwrap()]).unwrap();
    let s300 = run_counted(&single, &coordination(300));
    assert_eq!(s300.relations.subject.len(), 300 * 299);
    assert_eq!(s300.edges.from, s300.relations.subject);
    assert_eq!(s300.edges.to, s300.relations.object);
    assert!(last.unwrap().edges.score.iter().all(|&sc| sc == 0.9));
}

/// `n` sentences `"a b c. "` where `b` heads `a` (`nsubj`) and `c` (`obj`),
/// with entities on `a` and `c`: one match per sentence.
fn sentences(n: u32) -> Input {
    let text = "a b c. ".repeat(n as usize);
    let (mut spans, mut tokens, mut entity_tokens) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (at, t) = (7 * i, 3 * i);
        spans.push((at, at + 6));
        tokens.extend([(at, at + 1, Some(t + 1), 2), (at + 2, at + 3, None, 0), (at + 4, at + 5, Some(t + 1), 3)]);
        entity_tokens.extend([t, t + 2]);
    }
    input(&text, &spans, &tokens, &entity_tokens)
}

#[test]
fn pairs_and_heads_grow_linearly_with_the_document() {
    let annotator: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("r", "<nsubj >obj").unwrap()]).unwrap();
    let sizes = [4_000u32, 16_000, 64_000];
    let mut work_samples = Vec::new();
    let mut byte_samples = Vec::new();
    let mut call_samples = Vec::new();
    let mut last = None;
    for n in sizes {
        let s = run_counted(&annotator, &sentences(n));
        assert_eq!(s.relations.subject.len(), n as usize);
        work_samples.push((n as usize, s.work));
        byte_samples.push((n as usize, s.alloc_bytes));
        call_samples.push((n as usize, s.alloc_calls));
        last = Some(s);
    }
    RelationWork::check_linear_scaling(&work_samples).unwrap();
    assert_bytes_scale_linearly("sentences-bytes", &byte_samples);
    assert_bytes_scale_linearly("sentences-calls", &call_samples);

    let relations = last.unwrap().relations;
    assert_eq!(relations.subject.len(), 64_000);
    assert!(relations.subject.iter().zip(&relations.object).all(|(s, o)| s.get() % 2 == 0 && o.get() == s.get() + 1));
}

/// One sentence of `n` nested `ccomp` clauses (`v_0 -> v_1 -> ... -> v_{n-1}`),
/// each clause `i` holding `a_i <-nsubj- v_i -obj-> b_i`: `2n` entities in one
/// sentence, `n` matching pairs, and tree depth `n`.
fn deep_clauses(n: u32) -> Input {
    let text = "a b c ".repeat(n as usize);
    let (mut tokens, mut entity_tokens) = (Vec::new(), Vec::new());
    for i in 0..n {
        let (at, t) = (6 * i, 3 * i);
        let verb_head = (i > 0).then(|| 3 * (i - 1) + 1);
        let verb_rel = if i == 0 { 0 } else { 4 };
        tokens.extend([
            (at, at + 1, Some(t + 1), 2),
            (at + 2, at + 3, verb_head, verb_rel),
            (at + 4, at + 5, Some(t + 1), 3),
        ]);
        entity_tokens.extend([t, t + 2]);
    }
    input(&text, &[(0, text.len() as u32)], &tokens, &entity_tokens)
}

#[test]
fn deep_clause_chains_scale_without_root_walks() {
    let annotator: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("r", "<nsubj >obj").unwrap()]).unwrap();
    let mut work_samples = Vec::new();
    let mut byte_samples = Vec::new();
    let mut discover_bytes = Vec::new();
    let mut discover = RelationPathAnnotator::default();
    discover.max_path_len = Some(2);

    for clauses in [75u32, 150] {
        let inp = deep_clauses(clauses);
        let pairs = (2 * clauses as usize) * (2 * clauses as usize - 1);
        let s = run_counted(&annotator, &inp);
        assert_eq!(s.relations.subject.len(), clauses as usize);
        assert_eq!(s.edges.len(), clauses as usize);
        // Distant clause pairs (|i - j| >= 3) reject in O(1) with 0 arc reads;
        // same-clause (8n), adjacent-clause (8n - 8), and 2-apart pairs
        // (16n - 32) total exactly 32n - 40 arc reads instead of O(n^3).
        assert_eq!(s.work.arc_reads, 32 * clauses as usize - 40);
        assert!(s.alloc_calls < 512, "clauses={clauses}: expected < 512 alloc calls, got {}", s.alloc_calls);
        work_samples.push((pairs, s.work));
        byte_samples.push((clauses as usize, s.alloc_bytes));

        let view = inp.doc.get::<View>().unwrap();
        let b0 = BYTES.get();
        let (paths,) = discover
            .annotate((view, &inp.sentences, &inp.tokens, &inp.tokens, &inp.deps, &inp.entities, None, None))
            .unwrap();
        discover_bytes.push((clauses as usize, BYTES.get() - b0));
        assert_eq!(paths.len(), 2 * clauses as usize);
    }
    RelationWork::check_linear_scaling(&work_samples).unwrap();
    assert_bytes_scale_linearly("deep-clauses", &byte_samples);
    assert_bytes_scale_linearly("deep-clauses-discover", &discover_bytes);
}

#[test]
#[ignore = "manual wall-clock timing probe; run with --ignored --nocapture"]
fn wall_clock_timing_probe() {
    let coord: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("r", "<conj >conj").unwrap()]).unwrap();
    for n in [150u32, 300] {
        let inp = coordination(n);
        let start = Instant::now();
        let s = run_counted(&coord, &inp);
        println!("coordination({n}): {:?}, work={:?}, bytes={}", start.elapsed(), s.work, s.alloc_bytes);
    }
    let sents: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("r", "<nsubj >obj").unwrap()]).unwrap();
    for n in [16_000u32, 64_000] {
        let inp = sentences(n);
        let start = Instant::now();
        let s = run_counted(&sents, &inp);
        println!("sentences({n}): {:?}, work={:?}, bytes={}", start.elapsed(), s.work, s.alloc_bytes);
    }
}
