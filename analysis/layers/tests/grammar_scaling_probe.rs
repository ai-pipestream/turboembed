//! Linear scaling probe for `GrammarChecker` with dependency trees.
//!
//! Three measures, none of them wall-clock, so parallel `nextest` load cannot
//! cause false failures:
//!
//! - `dependency_lookup_scaling`: [`GrammarCheckWork`] counters on one long
//!   multi-clause sentence. The counters are taken at every read of token or
//!   dependency data, so a rule cannot do a slow lookup they miss
//!   (`src/quality/scaling_tests.rs` proves that with quadratic passes).
//! - `many_sentence_scaling`: the same counters on many short sentences, which
//!   covers the per-sentence index build.
//! - `allocation_scaling`: bytes and calls seen by a counting allocator. This
//!   does not depend on the checker reporting anything.
//!
//! Use `dependency_lookup_wall_clock_timing` with `--ignored --nocapture` for manual timing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;
use turbo_analysis::{Document, Ix, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, GrammarCheckWork, GrammarChecker, PosTags, Sentences, Tokens};

struct Counting;
thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}
fn count(bytes: usize) {
    CALLS.with(|c| c.set(c.get() + 1));
    BYTES.with(|b| b.set(b.get() + bytes));
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

const CLAUSE_COUNTS: [usize; 4] = [250, 500, 1000, 2000];

fn build_multi_clause_fixture(clauses: usize) -> (Document, Sentences, Tokens, PosTags, Dependencies) {
    let text = "They work and ".repeat(clauses);
    let document = Document::new(text.clone());
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    sentence_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: sentence_spans };
    let mut spans = Spans::new(ViewId::SOURCE);
    let tagset = TagSet::new("penn", &["PRP", "VBP", "CC"]);
    let relations = TagSet::new("ud", &["nsubj", "root", "cc", "conj"]);
    let mut tags = Vec::new();
    let mut head = Vec::new();
    let mut relation = Vec::new();
    for k in 0..clauses {
        for (offset, len, tag) in [(0, 4, "PRP"), (5, 4, "VBP"), (10, 3, "CC")] {
            spans.push(Span::new((k * 14 + offset) as u32, (k * 14 + offset + len) as u32));
            tags.push(tagset.tag(tag).unwrap());
        }
        head.extend([
            Some(Ix::new((3 * k + 1) as u32)),
            if k == 0 { None } else { Some(Ix::new(1)) },
            Some(Ix::new((3 * k + 1) as u32)),
        ]);
        relation.extend([
            relations.tag("nsubj").unwrap(),
            relations.tag(if k == 0 { "root" } else { "conj" }).unwrap(),
            relations.tag("cc").unwrap(),
        ]);
    }
    let tokens = Tokens { spans, sentence: None };
    let pos = PosTags { tagset, tags, prob: None };
    let deps = Dependencies { head, relation, relations };
    (document, sentences, tokens, pos, deps)
}

/// `They work. ` repeated: one three-token sentence per repetition.
fn build_many_sentence_fixture(count: usize) -> (Document, Sentences, Tokens, PosTags, Dependencies) {
    let text = "They work. ".repeat(count);
    let document = Document::new(text);
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    let mut spans = Spans::new(ViewId::SOURCE);
    let tagset = TagSet::new("penn", &["PRP", "VBP", "."]);
    let relations = TagSet::new("ud", &["nsubj", "root", "punct"]);
    let (mut tags, mut head, mut relation) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..count {
        let at = (k * 11) as u32;
        sentence_spans.push(Span::new(at, at + 10));
        for (offset, len, tag) in [(0, 4, "PRP"), (5, 4, "VBP"), (9, 1, ".")] {
            spans.push(Span::new(at + offset, at + offset + len));
            tags.push(tagset.tag(tag).unwrap());
        }
        let verb = Some(Ix::new((3 * k + 1) as u32));
        head.extend([verb, None, verb]);
        for rel in ["nsubj", "root", "punct"] {
            relation.push(relations.tag(rel).unwrap());
        }
    }
    let tokens = Tokens { spans, sentence: None };
    let pos = PosTags { tagset, tags, prob: None };
    let deps = Dependencies { head, relation, relations };
    (document, Sentences { spans: sentence_spans }, tokens, pos, deps)
}

type Fixture = fn(usize) -> (Document, Sentences, Tokens, PosTags, Dependencies);

fn work_by_size(label: &str, fixture: Fixture) -> Vec<(usize, GrammarCheckWork)> {
    let checker = GrammarChecker::new();
    CLAUSE_COUNTS
        .iter()
        .map(|&count| {
            let (document, sentences, tokens, pos, deps) = fixture(count);
            let view = document.get::<View>().unwrap();
            let (diagnostics, work) =
                checker.check_view_with_work(view, &sentences, &tokens, &pos, None, Some(&deps)).unwrap();
            assert!(diagnostics.is_empty(), "expected clean fixture, got {} diagnostics", diagnostics.len());
            assert!(work.dep_lookups >= count * 3, "every head is read at least once: {work:?}");
            assert!(work.token_reads >= count * 3, "every token is read at least once: {work:?}");
            println!("{label} tokens={} token_reads={} dep_lookups={}", count * 3, work.token_reads, work.dep_lookups);
            (count * 3, work)
        })
        .collect()
}

#[test]
fn dependency_lookup_scaling() {
    let samples = work_by_size("one-sentence", build_multi_clause_fixture);
    GrammarCheckWork::check_linear_scaling(&samples).unwrap();
}

#[test]
fn many_sentence_scaling() {
    let samples = work_by_size("many-sentences", build_many_sentence_fixture);
    GrammarCheckWork::check_linear_scaling(&samples).unwrap();
}

#[test]
fn allocation_scaling() {
    let checker = GrammarChecker::new();
    for (label, fixture) in
        [("one-sentence", build_multi_clause_fixture as Fixture), ("many-sentences", build_many_sentence_fixture)]
    {
        let mut samples = Vec::new();
        for count in CLAUSE_COUNTS {
            let (document, sentences, tokens, pos, deps) = fixture(count);
            let view = document.get::<View>().unwrap();
            let (calls, bytes) = (CALLS.get(), BYTES.get());
            let result = checker.check_view(view, &sentences, &tokens, &pos, None, Some(&deps)).unwrap();
            let (calls, bytes) = (CALLS.get() - calls, BYTES.get() - bytes);
            assert!(result.is_empty());
            println!("{label} tokens={} alloc_calls={calls} alloc_bytes={bytes}", count * 3);
            samples.push((count * 3, calls, bytes));
        }
        for pair in samples.windows(2) {
            let ((n_prev, calls_prev, bytes_prev), (n_curr, calls_curr, bytes_curr)) = (pair[0], pair[1]);
            for (name, prev, curr) in [("calls", calls_prev, calls_curr), ("bytes", bytes_prev, bytes_curr)] {
                assert!(
                    curr * 100 <= prev * 225,
                    "{label}: allocation {name} grew faster than linear when tokens doubled \
                     ({n_prev} -> {n_curr}): {prev} -> {curr}"
                );
            }
        }
    }
}

#[test]
#[ignore = "manual wall-clock timing probe; run with --ignored --nocapture"]
fn dependency_lookup_wall_clock_timing() {
    let checker = GrammarChecker::new();
    let mut medians = Vec::new();
    for clauses in CLAUSE_COUNTS {
        let (document, sentences, tokens, pos, deps) = build_multi_clause_fixture(clauses);
        let view = document.get::<View>().unwrap();
        let mut timings = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let result = checker.check_view(view, &sentences, &tokens, &pos, None, Some(&deps)).unwrap();
            std::hint::black_box(result);
            timings.push(start.elapsed().as_nanos());
        }
        timings.sort_unstable();
        println!("tokens={} median_ns={}", clauses * 3, timings[2]);
        medians.push((clauses * 3, timings[2]));
    }
    let (_, t_750) = medians[0];
    let (_, t_6000) = medians[3];
    assert!(
        t_6000 < t_750.saturating_mul(25),
        "8x token increase should scale near-linearly, got 750 tokens = {t_750}ns vs 6000 tokens = {t_6000}ns"
    );
}
