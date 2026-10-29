//! A compiled morphology set test allocates nothing per token (#1503).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::{MorphFeatures, Strs, Tokens};
use turbo_matcher::{Input, Matcher, spacy::read_pattern};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocations one `find` makes over `n` tokens, a quarter of which
/// pass every set test.
fn allocations(m: &Matcher, n: usize) -> usize {
    let feats = ["Number=Sing|Case=Nom,Acc", "_", "Case=Nom", "Tense=Past|VerbForm=Fin"];
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut strs = Strs::default();
    for i in 0..n {
        let start = text.len() as u32;
        text.push_str("w ");
        spans.push(Span::new(start, start + 1));
        strs.push(feats[i % feats.len()]);
    }
    let tokens = Tokens { spans, sentence: None };
    let morph = MorphFeatures::new(strs);
    let input = Input::new(&text, &tokens).unwrap().with_morph(&morph).unwrap();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let hits = m.find(&input).unwrap();
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    assert!(hits.len() >= n / 4, "{} hits over {n} tokens", hits.len());
    after - before
}

#[test]
fn set_tests_allocate_nothing_per_token() {
    let mut m = Matcher::new();
    for (label, pattern) in [
        ("superset", r#"[{"MORPH":{"IS_SUPERSET":["Case=Acc"]}}]"#),
        ("subset", r#"[{"MORPH":{"IS_SUBSET":["Number=Sing","Case=Nom","Case=Acc"]}}]"#),
        ("intersects", r#"[{"MORPH":{"INTERSECTS":["Case=Acc","Mood=Ind"]}}]"#),
    ] {
        m.add(label, &read_pattern(pattern).unwrap()).unwrap();
    }
    let small = allocations(&m, 1_000);
    let large = allocations(&m, 10_000);
    // The match list and columns grow by doubling, so ten times the tokens
    // add a few dozen allocations at most, not one or more per token.
    assert!(large.saturating_sub(small) < 200, "{small} allocations for 1,000 tokens, {large} for 10,000");
}
