//! Proof that the grammar scaling check catches slow lookups.
//!
//! Each case runs the real sentence rules and then one extra pass over the
//! same [`SentCtx`], written the way a slow rule would be written: for every
//! verb it rescans the whole sentence. The pass carries no counting code. The
//! check must reject every such pass and must accept the real rules alone and
//! the real rules plus a linear pass.

use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, TagSet, ViewId};

use crate::{Dependencies, PosTags, Tokens};

use super::agreement::run_on_sentence;
use super::grammar::GrammarChecker;
use super::sent_ctx::SentCtx;
use super::work::GrammarCheckWork;

const CLAUSE_COUNTS: [usize; 4] = [250, 500, 1000, 2000];

struct Fixture {
    text: String,
    tokens: Tokens,
    pos: PosTags,
    deps: Dependencies,
}

/// `They work and ` repeated: every verb hangs off the first verb.
fn fixture(clauses: usize) -> Fixture {
    let text = "They work and ".repeat(clauses);
    let mut spans = Spans::new(ViewId::SOURCE);
    let tagset = TagSet::new("penn", &["PRP", "VBP", "CC"]);
    let relations = TagSet::new("ud", &["nsubj", "root", "cc", "conj"]);
    let (mut tags, mut head, mut relation) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..clauses {
        for (offset, len, tag) in [(0, 4, "PRP"), (5, 4, "VBP"), (10, 3, "CC")] {
            spans.push(Span::new((k * 14 + offset) as u32, (k * 14 + offset + len) as u32));
            tags.push(tagset.tag(tag).unwrap());
        }
        let verb = Some(Ix::new((3 * k + 1) as u32));
        head.extend([verb, if k == 0 { None } else { Some(Ix::new(1)) }, verb]);
        relation.extend([
            relations.tag("nsubj").unwrap(),
            relations.tag(if k == 0 { "root" } else { "conj" }).unwrap(),
            relations.tag("cc").unwrap(),
        ]);
    }
    Fixture {
        text,
        tokens: Tokens { spans, sentence: None },
        pos: PosTags { tagset, tags, prob: None },
        deps: Dependencies { head, relation, relations },
    }
}

/// Runs the real rules, then `extra`, on one sentence per size.
fn work_with(extra: impl Fn(&SentCtx<'_>)) -> Vec<(usize, GrammarCheckWork)> {
    let checker = GrammarChecker::new();
    CLAUSE_COUNTS
        .iter()
        .map(|&clauses| {
            let f = fixture(clauses);
            let range = IxRange::new(0, f.tokens.len() as u32);
            let cx = SentCtx::new(&checker, &f.text, &f.tokens, &f.pos, None, Some(&f.deps), None, range);
            let mut out = Vec::new();
            run_on_sentence(&cx, &mut out);
            assert!(out.is_empty(), "fixture must be clean, got {} diagnostics", out.len());
            extra(&cx);
            (clauses * 3, cx.work())
        })
        .collect()
}

/// Calls `per_pair(verb, token)` for every verb and every token of the sentence.
fn for_each_verb_rescan(cx: &SentCtx<'_>, per_pair: impl Fn(usize, usize) -> bool) {
    let (start, end) = (cx.range.start.get(), cx.range.end.get());
    let mut hits = 0usize;
    for v in (start..end).filter(|v| v % 3 == 1) {
        hits += (start..end).filter(|&t| per_pair(v, t)).count();
    }
    std::hint::black_box(hits);
}

fn assert_caught(name: &str, samples: &[(usize, GrammarCheckWork)]) {
    let verdict = GrammarCheckWork::check_linear_scaling(samples);
    assert!(verdict.is_err(), "{name}: a quadratic pass went unseen, counters were {samples:?}");
}

#[test]
fn real_rules_alone_pass_the_check() {
    let samples = work_with(|_| {});
    GrammarCheckWork::check_linear_scaling(&samples).unwrap();
    assert!(samples.iter().all(|(_, w)| w.dep_lookups > 0), "{samples:?}");
}

#[test]
fn a_linear_extra_pass_still_passes_the_check() {
    let samples = work_with(|cx| {
        let hits = (cx.range.start.get()..cx.range.end.get()).filter(|&t| cx.surface(t) == "work").count();
        assert!(hits > 0);
    });
    GrammarCheckWork::check_linear_scaling(&samples).unwrap();
}

#[test]
fn rescanning_dependency_heads_per_verb_is_caught() {
    // The reviewer's probe on #2382: children found by scanning every head.
    assert_caught("dep_head", &work_with(|cx| for_each_verb_rescan(cx, |v, t| cx.dep_head(t) == Some(v))));
}

#[test]
fn rescanning_dependency_relations_per_verb_is_caught() {
    assert_caught("dep_rel", &work_with(|cx| for_each_verb_rescan(cx, |_, t| cx.dep_rel(t) == Some("nsubj"))));
}

#[test]
fn rescanning_surfaces_per_verb_is_caught() {
    // An unbounded subject window: the fallback path without its 24 token cap.
    assert_caught("surface", &work_with(|cx| for_each_verb_rescan(cx, |_, t| cx.surface(t) == "They")));
}

#[test]
fn rescanning_tags_per_verb_is_caught() {
    assert_caught("pos_str", &work_with(|cx| for_each_verb_rescan(cx, |_, t| cx.pos_str(t) == "PRP")));
}

#[test]
fn rescanning_spans_per_verb_is_caught() {
    assert_caught("span", &work_with(|cx| for_each_verb_rescan(cx, |v, t| cx.span(t).end <= cx.span(v).start)));
}

#[test]
fn rescanning_gaps_per_verb_is_caught() {
    assert_caught(
        "gap_after",
        &work_with(|cx| for_each_verb_rescan(cx, |_, t| t + 1 < cx.range.end.get() && cx.gap_after(t) == " ")),
    );
}

#[test]
fn rescanning_lemmas_and_morphology_per_verb_is_caught() {
    assert_caught("lemma_str", &work_with(|cx| for_each_verb_rescan(cx, |_, t| cx.lemma_str(t).is_some())));
    assert_caught("morph_str", &work_with(|cx| for_each_verb_rescan(cx, |_, t| cx.morph_str(t).is_some())));
}

#[test]
fn walking_the_root_child_list_per_verb_is_caught() {
    // Token 1 heads every other verb, so its child list grows with the sentence.
    assert_caught(
        "dep_children",
        &work_with(|cx| {
            let (start, end) = (cx.range.start.get(), cx.range.end.get());
            let seen: usize = (start..end).filter(|v| v % 3 == 1).map(|_| cx.dep_children(1).len()).sum();
            assert!(seen > 0);
        }),
    );
}

#[test]
fn index_build_counts_each_head_it_visits() {
    let checker = GrammarChecker::new();
    let f = fixture(250);
    let n = f.tokens.len();
    let range = IxRange::new(0, n as u32);
    let with_deps = SentCtx::new(&checker, &f.text, &f.tokens, &f.pos, None, Some(&f.deps), None, range);
    assert_eq!(with_deps.work(), GrammarCheckWork { token_reads: 0, dep_lookups: n });
    let without = SentCtx::new(&checker, &f.text, &f.tokens, &f.pos, None, None, None, range);
    assert_eq!(without.work(), GrammarCheckWork::default());
}

#[test]
fn indexing_the_whole_document_for_every_sentence_is_caught() {
    // What a per-sentence index build would cost if it walked every head in
    // the document instead of only the heads in its own sentence.
    let checker = GrammarChecker::new();
    let samples: Vec<_> = CLAUSE_COUNTS
        .iter()
        .map(|&clauses| {
            let f = fixture(clauses);
            let whole = IxRange::new(0, f.tokens.len() as u32);
            let mut work = GrammarCheckWork::default();
            for _sentence in 0..clauses {
                work.add(SentCtx::new(&checker, &f.text, &f.tokens, &f.pos, None, Some(&f.deps), None, whole).work());
            }
            (clauses * 3, work)
        })
        .collect();
    assert_caught("index build", &samples);
}

#[test]
fn check_rejects_samples_that_cannot_show_scaling() {
    let w = GrammarCheckWork { token_reads: 10, dep_lookups: 10 };
    assert!(GrammarCheckWork::check_linear_scaling(&[(100, w)]).is_err());
    assert!(GrammarCheckWork::check_linear_scaling(&[(100, w), (300, w)]).is_err());
    let doubled = GrammarCheckWork { token_reads: 20, dep_lookups: 20 };
    GrammarCheckWork::check_linear_scaling(&[(100, w), (200, doubled)]).unwrap();
    // One counter flat, the other quadratic: the flat one must not hide it.
    let skewed = GrammarCheckWork { token_reads: 20, dep_lookups: 40 };
    let err = GrammarCheckWork::check_linear_scaling(&[(100, w), (200, skewed)]).unwrap_err();
    assert!(err.contains("dep_lookups"), "{err}");
}
