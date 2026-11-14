//! Linear scaling probe for `GrammarChecker` with dependency trees on long multi-clause sentences.
//!
//! `dependency_lookup_scaling` asserts on deterministic [`GrammarCheckWork`] counters
//! rather than wall-clock time so parallel `nextest` load cannot cause false failures.
//! Use `dependency_lookup_wall_clock_timing` with `--ignored --nocapture` for manual timing.

use std::time::Instant;
use turbo_analysis::{Document, Ix, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, GrammarCheckWork, GrammarChecker, PosTags, Sentences, Tokens};

fn build_multi_clause_fixture(clauses: usize) -> (Document, Sentences, Tokens, PosTags, Dependencies) {
    let text = std::iter::repeat_n("They work and ", clauses).collect::<String>();
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

#[test]
fn dependency_lookup_scaling() {
    let checker = GrammarChecker::new();
    let mut work_by_size: Vec<(usize, GrammarCheckWork)> = Vec::new();
    for clauses in [250, 500, 1000, 2000] {
        let (document, sentences, tokens, pos, deps) = build_multi_clause_fixture(clauses);
        let view = document.get::<View>().unwrap();
        let (diagnostics, work) =
            checker.check_view_with_work(view, &sentences, &tokens, &pos, None, Some(&deps)).unwrap();
        assert!(diagnostics.is_empty(), "expected clean fixture, got {} diagnostics", diagnostics.len());
        assert!(work.dep_lookups > 0, "expected dependency lookups to be recorded");
        assert!(work.tokens_visited > 0, "expected visited tokens to be recorded");
        println!(
            "tokens={} dep_lookups={} tokens_visited={} window_scans={} total_units={}",
            clauses * 3,
            work.dep_lookups,
            work.tokens_visited,
            work.window_scans,
            work.total_units()
        );
        work_by_size.push((clauses * 3, work));
    }

    for pair in work_by_size.windows(2) {
        let (n_prev, w_prev) = pair[0];
        let (n_curr, w_curr) = pair[1];
        assert!(
            w_curr.dep_lookups <= w_prev.dep_lookups.saturating_mul(3),
            "2x token increase ({n_prev} -> {n_curr}) should scale linearly in dep_lookups: {} -> {}",
            w_prev.dep_lookups,
            w_curr.dep_lookups
        );
        assert!(
            w_curr.total_units() <= w_prev.total_units().saturating_mul(3),
            "2x token increase ({n_prev} -> {n_curr}) should scale linearly in total_units: {} -> {}",
            w_prev.total_units(),
            w_curr.total_units()
        );
    }

    let (_, w_750) = work_by_size[0];
    let (_, w_6000) = work_by_size[3];
    assert!(
        w_6000.dep_lookups <= w_750.dep_lookups.saturating_mul(10),
        "8x token increase should scale linearly in dep_lookups: 750 tokens = {} vs 6000 tokens = {}",
        w_750.dep_lookups,
        w_6000.dep_lookups
    );
    assert!(
        w_6000.total_units() <= w_750.total_units().saturating_mul(10),
        "8x token increase should scale linearly in total_units: 750 tokens = {} vs 6000 tokens = {}",
        w_750.total_units(),
        w_6000.total_units()
    );
}

#[test]
#[ignore = "manual wall-clock timing probe; run with --ignored --nocapture"]
fn dependency_lookup_wall_clock_timing() {
    let checker = GrammarChecker::new();
    let mut medians = Vec::new();
    for clauses in [250, 500, 1000, 2000] {
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
